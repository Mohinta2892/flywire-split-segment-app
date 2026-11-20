"""Backend service orchestrating CATMAID <-> CAVE bidirectional proofreading, exports, and audit logging."""
import contextlib
import csv
import inspect
import io
import json
import re
import threading
import time
import urllib.parse
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import navis
import numpy as np
import pandas as pd
import pymaid
from caveclient import CAVEclient

from catmaid_cave_sync.export import build_cave_export_plan, execute_cave_export_plan
from catmaid_cave_sync.export.catmaid_matcher import _get_neurons_fast, _no_pymaid_cache
from catmaid_cave_sync.export.retry import retry_call
from catmaid_cave_sync.spatial_mapper import map_nodes_to_cave


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _parse_int_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        tokens = [str(x).strip() for x in raw]
    else:
        tokens = re.split(r"[\s,;]+", str(raw).strip())
    out: list[int] = []
    for tok in tokens:
        if not tok:
            continue
        val = int(tok)
        if val not in out:
            out.append(val)
    return out


def _parse_res_triplet(raw: Any, label: str) -> list[float]:
    if isinstance(raw, (list, tuple)):
        vals = [float(x) for x in raw]
    else:
        vals = [float(x.strip()) for x in str(raw or "").split(",") if x.strip()]
    if len(vals) != 3 or any(v <= 0 or np.isnan(v) for v in vals):
        raise ValueError(f"Invalid {label}: expected 3 positive numbers (x,y,z nm).")
    return vals


class CatmaidService:
    """Thread-safe controller for CATMAID <-> CAVE syncing, dry-run planning, execution, and changelog."""

    def __init__(self, app_config: dict | None = None, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
        self.app_config = app_config if app_config is not None else self._load_app_config()
        self.proj_cfg_path = self.base_dir / ".cave_catmaid_config.json"
        self.outputs_dir = self.base_dir / "outputs"
        self.cache_dir = self.outputs_dir / "catmaid_swc_cache"
        self.changelog_json_path = self.outputs_dir / "change_register.json"
        self.changelog_csv_path = self.outputs_dir / "change_register.csv"

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._projects: dict[str, dict] = self._load_projects()
        self._active_project: str | None = next(iter(self._projects.keys()), None)
        self._changelog: list[dict] = self._load_changelog()

        # CATMAID -> CAVE state
        self._status = "idle"
        self._progress = ""
        self._message = ""
        self._summary: dict | None = None
        self._splits: list[dict] = []
        self._merges: list[dict] = []
        self._audit: list[dict] = []
        self._plan_project: str | None = None

        # CAVE -> CATMAID export state
        self._export_status = "idle"
        self._export_progress = ""
        self._export_message = ""
        self._export_pct = 0.0
        self._export_started_at: float | None = None
        self._export_elapsed_sec: float | None = None
        self._export_plan: list[dict] = []
        self._export_bundle: dict | None = None
        self._export_audit: list[dict] = []
        self._export_project: str | None = None

    def _load_app_config(self) -> dict:
        cfg_file = self.base_dir / "config.json"
        if cfg_file.exists():
            try:
                return json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _sample_datastacks(self) -> tuple[dict[str, str], str]:
        samples_raw = self.app_config.get("samples", {})
        default_srv = str(
            self.app_config.get("server_address")
            or self.app_config.get("default_server")
            or self.app_config.get("cave_server_address")
            or "https://global.daf-apis.com"
        )
        mapping: dict[str, str] = {}
        if isinstance(samples_raw, dict):
            for k, v in samples_raw.items():
                if isinstance(v, dict):
                    ds = v.get("datastack") or v.get("cave_datastack") or ""
                    if ds:
                        mapping[str(k)] = str(ds)
                elif isinstance(v, str) and v:
                    mapping[str(k)] = v
        elif isinstance(samples_raw, list):
            for item in samples_raw:
                if isinstance(item, dict) and item.get("name") and item.get("datastack"):
                    mapping[str(item["name"])] = str(item["datastack"])
        return mapping, default_srv

    def _load_projects(self) -> dict[str, dict]:
        if not self.proj_cfg_path.exists():
            return {}
        try:
            raw = json.loads(self.proj_cfg_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw.get("projects", raw) if "projects" in raw and isinstance(raw["projects"], dict) else raw
        except Exception:
            pass
        return {}

    def _save_projects(self) -> None:
        self.proj_cfg_path.write_text(json.dumps(self._projects, indent=2), encoding="utf-8")

    def _load_changelog(self) -> list[dict]:
        if not self.changelog_json_path.exists():
            return []
        try:
            data = json.loads(self.changelog_json_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _persist_changelog(self) -> None:
        self.changelog_json_path.write_text(json.dumps(self._changelog, indent=2), encoding="utf-8")
        fieldnames = [
            "id",
            "timestamp",
            "system",
            "project",
            "target",
            "action",
            "before_ids",
            "after_ids",
            "operation_id",
            "summary",
        ]
        with self.changelog_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._changelog:
                writer.writerow(row)

    def _record_change(
        self,
        system: str,
        project: str,
        action: str,
        before_ids: str,
        after_ids: str,
        operation_id: str,
        summary: str,
        target: str = "",
    ) -> None:
        with self._lock:
            next_id = (max((int(x.get("id", 0)) for x in self._changelog), default=0)) + 1
            entry = {
                "id": next_id,
                "timestamp": _utc_now_str(),
                "system": system,
                "project": project,
                "target": target,
                "action": action,
                "before_ids": str(before_ids),
                "after_ids": str(after_ids),
                "operation_id": str(operation_id or ""),
                "summary": summary,
            }
            self._changelog.insert(0, entry)
            self._persist_changelog()

    def changelog_csv(self) -> str:
        with self._lock:
            if self.changelog_csv_path.exists():
                return self.changelog_csv_path.read_text(encoding="utf-8")
            buf = io.StringIO()
            fieldnames = [
                "id",
                "timestamp",
                "system",
                "project",
                "target",
                "action",
                "before_ids",
                "after_ids",
                "operation_id",
                "summary",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._changelog:
                writer.writerow(row)
            return buf.getvalue()

    def _build_cave_client(self, cfg: dict) -> CAVEclient:
        ds = str(cfg.get("cave_datastack") or "").strip()
        srv = str(cfg.get("cave_server_address") or "").strip()
        if not ds:
            raise ValueError("CAVE Datastack is required.")
        return CAVEclient(datastack_name=ds, server_address=srv or None)

    def _build_catmaid_instance(self, cfg: dict) -> pymaid.CatmaidInstance:
        url = str(cfg.get("catmaid_url") or "").strip().split("?")[0].rstrip("/")
        pid = int(cfg.get("catmaid_project_id") or 0)
        token = str(cfg.get("catmaid_token") or "").strip()
        http_user = str(cfg.get("catmaid_http_user") or "").strip()
        http_pass = str(cfg.get("catmaid_http_password") or "").strip()
        if not url or pid <= 0:
            raise ValueError("CATMAID Server URL and Project ID are required.")
        if not token:
            raise ValueError("CATMAID API Token is required.")
        kw: dict[str, Any] = {
            "server": url,
            "api_token": token,
            "project_id": pid,
        }
        sig = inspect.signature(pymaid.CatmaidInstance)
        if "http_user" in sig.parameters and http_user:
            kw["http_user"] = http_user
        if "http_password" in sig.parameters and http_pass:
            kw["http_password"] = http_pass
        if "caching" in sig.parameters:
            kw["caching"] = False
        return pymaid.CatmaidInstance(**kw)

    def _get_project_cfg(self, project_name: str) -> dict:
        with self._lock:
            cfg = self._projects.get(project_name)
            if not cfg:
                raise ValueError(f"Project profile '{project_name}' is not configured yet.")
            return dict(cfg)

    def public_state(self) -> dict:
        sample_ds, default_srv = self._sample_datastacks()
        with self._lock:
            sanitized_projects: dict[str, dict] = {}
            for name, cfg in self._projects.items():
                sanitized_projects[name] = {
                    "cave_datastack": cfg.get("cave_datastack", ""),
                    "cave_server_address": cfg.get("cave_server_address", default_srv),
                    "cave_resolution_nm": cfg.get("cave_resolution_nm", []),
                    "catmaid_url": cfg.get("catmaid_url", ""),
                    "catmaid_project_id": cfg.get("catmaid_project_id", 15),
                    "catmaid_http_user": cfg.get("catmaid_http_user", ""),
                    "catmaid_resolution_nm": cfg.get("catmaid_resolution_nm", []),
                    "catmaid_coord_unit": cfg.get("catmaid_coord_unit", "nm"),
                    "has_token": bool(cfg.get("catmaid_token")),
                }

            running_sec = None
            if self._export_status in ("loading", "executing") and self._export_started_at is not None:
                running_sec = round(max(0.0, time.time() - self._export_started_at), 2)

            return {
                "sample_datastacks": sample_ds,
                "default_server": default_srv,
                "projects": sanitized_projects,
                "active_project": self._active_project,
                "status": self._status,
                "progress": self._progress,
                "message": self._message,
                "summary": self._summary,
                "splits": self._splits,
                "merges": self._merges,
                "audit": self._audit,
                "export_status": self._export_status,
                "export_progress": self._export_progress,
                "export_message": self._export_message,
                "export_pct": round(self._export_pct, 1),
                "export_running_sec": running_sec,
                "export_elapsed_sec": self._export_elapsed_sec,
                "export_plan": self._export_plan,
                "export_audit": self._export_audit,
                "changelog": list(self._changelog),
            }

    def probe_resolutions(self, d: dict) -> dict:
        proj_name = str(d.get("project_name") or "").strip()
        existing = self._projects.get(proj_name, {}) if proj_name else {}
        merged = {
            "cave_datastack": d.get("cave_datastack") or existing.get("cave_datastack", ""),
            "cave_server_address": d.get("cave_server_address") or existing.get("cave_server_address", ""),
            "catmaid_url": d.get("catmaid_url") or existing.get("catmaid_url", ""),
            "catmaid_project_id": d.get("catmaid_project_id") or existing.get("catmaid_project_id", 15),
            "catmaid_token": d.get("catmaid_token") or existing.get("catmaid_token", ""),
            "catmaid_http_user": d.get("catmaid_http_user") or existing.get("catmaid_http_user", ""),
            "catmaid_http_password": (
                d.get("catmaid_http_password")
                or d.get("catmaid_http_pass")
                or existing.get("catmaid_http_password", "")
            ),
        }

        cave_client = self._build_cave_client(merged)
        cv_sig = inspect.signature(cave_client.info.segmentation_cloudvolume)
        sink = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            warnings.simplefilter("ignore")
            cv = (
                cave_client.info.segmentation_cloudvolume(progress=False)
                if "progress" in cv_sig.parameters
                else cave_client.info.segmentation_cloudvolume()
            )
        cave_res = [float(x) for x in cv.resolution]

        rm = self._build_catmaid_instance(merged)
        stacks_out: list[dict] = []
        with _no_pymaid_cache(rm):
            stacks_url = rm.make_url(rm.project_id, "stacks")
            raw_stacks = rm.fetch(stacks_url)
            if isinstance(raw_stacks, list):
                for s in raw_stacks:
                    sid = s.get("id") or s.get("stack_id")
                    title = s.get("title") or f"Stack {sid}"
                    res_nm = None
                    try:
                        info = rm.fetch(rm.make_url(rm.project_id, "stack", sid, "info"))
                        r = info.get("resolution") if isinstance(info, dict) else None
                        if isinstance(r, dict) and all(k in r for k in ("x", "y", "z")):
                            res_nm = [float(r["x"]), float(r["y"]), float(r["z"])]
                        elif isinstance(r, (list, tuple)) and len(r) >= 3:
                            res_nm = [float(r[0]), float(r[1]), float(r[2])]
                    except Exception:
                        pass
                    stacks_out.append({
                        "stack_id": sid,
                        "title": title,
                        "resolution_nm": res_nm,
                    })

        return {
            "cave_resolution_nm": cave_res,
            "catmaid_stacks": stacks_out,
        }

    def save_project(self, d: dict) -> dict:
        name = str(d.get("project_name") or "").strip()
        if not name or name == "+ Custom Project":
            raise ValueError("Enter a valid Project Name before saving.")

        with self._lock:
            existing = dict(self._projects.get(name, {}))
            token = str(d.get("catmaid_token") or "").strip() or existing.get("catmaid_token", "")
            http_pass = (
                str(d.get("catmaid_http_password") or d.get("catmaid_http_pass") or "").strip()
                or existing.get("catmaid_http_password", "")
            )
            if not token:
                raise ValueError("CATMAID API Token is required.")

            cm_res = _parse_res_triplet(d.get("catmaid_resolution_nm"), "CATMAID Resolution")
            cave_res = _parse_res_triplet(d.get("cave_resolution_nm"), "CAVE Resolution")
            unit = str(d.get("catmaid_coord_unit") or "nm").strip().lower()
            if unit not in ("nm", "voxels"):
                unit = "nm"

            self._projects[name] = {
                "cave_datastack": str(d.get("cave_datastack") or "").strip(),
                "cave_server_address": str(d.get("cave_server_address") or "").strip(),
                "cave_resolution_nm": cave_res,
                "catmaid_url": str(d.get("catmaid_url") or "").strip().split("?")[0].rstrip("/"),
                "catmaid_project_id": int(d.get("catmaid_project_id") or 15),
                "catmaid_token": token,
                "catmaid_http_user": str(d.get("catmaid_http_user") or "").strip(),
                "catmaid_http_password": http_pass,
                "catmaid_resolution_nm": cm_res,
                "catmaid_coord_unit": unit,
            }
            self._active_project = name
            self._save_projects()
        return {"ok": True, "project_name": name}

    def select_project(self, d: dict) -> dict:
        name = str(d.get("project_name") or "").strip()
        with self._lock:
            if name in self._projects:
                self._active_project = name
        return {"ok": True, "active_project": self._active_project}

    def delete_project(self, d: dict) -> dict:
        name = str(d.get("project_name") or "").strip()
        with self._lock:
            if name in self._projects:
                del self._projects[name]
                if self._active_project == name:
                    self._active_project = next(iter(self._projects.keys()), None)
                self._save_projects()
        return {"ok": True}

    # -------------------------------------------------------------------------
    # 1. CATMAID -> CAVE Proofreading Plan, 3D Split Preview, & Execution
    # -------------------------------------------------------------------------

    def start_plan(self, d: dict) -> dict:
        proj = str(d.get("project_name") or "").strip()
        cfg = self._get_project_cfg(proj)
        mode = str(d.get("mode") or "annotation").strip().lower()
        query = str(d.get("query") or "").strip()
        min_nodes = max(1, int(d.get("min_nodes") or 3))
        force_refresh = bool(d.get("force_refresh", False))
        if not query:
            raise ValueError("Enter a CATMAID annotation or comma-separated skeleton IDs.")

        with self._lock:
            if self._status in ("loading", "executing"):
                raise RuntimeError("A CATMAID -> CAVE job is already running.")
            self._status = "loading"
            self._progress = "Connecting to CATMAID & CAVE..."
            self._message = ""
            self._summary = None
            self._splits = []
            self._merges = []
            self._audit = []
            self._plan_project = proj
            self._active_project = proj

        t = threading.Thread(
            target=self._run_plan_worker,
            args=(proj, cfg, mode, query, min_nodes, force_refresh),
            daemon=True,
        )
        t.start()
        return {"ok": True}

    def _run_plan_worker(
        self,
        proj: str,
        cfg: dict,
        mode: str,
        query: str,
        min_nodes: int,
        force_refresh: bool,
    ) -> None:
        def on_progress(msg: str, _pct: float | None = None) -> None:
            with self._lock:
                self._progress = str(msg)

        try:
            rm = self._build_catmaid_instance(cfg)
            cave_client = self._build_cave_client(cfg)
            cm_res = tuple(float(x) for x in cfg["catmaid_resolution_nm"])
            cave_res = tuple(float(x) for x in cfg["cave_resolution_nm"])
            unit = cfg.get("catmaid_coord_unit", "nm")
            scale = np.asarray(cm_res, dtype=np.float64) if unit == "voxels" else np.ones(3, dtype=np.float64)

            with _no_pymaid_cache(rm):
                if mode == "skids":
                    skids = _parse_int_list(query)
                    if not skids:
                        raise ValueError("No valid numeric skeleton IDs provided.")
                else:
                    anns = [a.strip() for a in query.split(",") if a.strip()]
                    skids = list(map(int, pymaid.get_skids_by_annotation(anns, remote_instance=rm)))
                    if not skids:
                        raise RuntimeError(f"No CATMAID skeletons found for annotation(s): {query}")

                on_progress(f"Fetching {len(skids)} CATMAID skeleton(s)...")

                proj_cache = self.cache_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", proj)
                proj_cache.mkdir(parents=True, exist_ok=True)
                neurons: list[navis.TreeNeuron] = []
                to_fetch: list[int] = []

                for s in skids:
                    swc_file = proj_cache / f"{int(s)}.swc"
                    if not force_refresh and swc_file.exists():
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                nrn = navis.read_swc(str(swc_file))
                            nrn.id = int(s)
                            neurons.append(nrn)
                            continue
                        except Exception:
                            pass
                    to_fetch.append(int(s))

                if to_fetch:
                    fetched = _get_neurons_fast(rm, to_fetch, batch_size=40, progress_callback=on_progress)
                    for nrn in fetched:
                        neurons.append(nrn)
                        try:
                            navis.write_swc(nrn, str(proj_cache / f"{int(nrn.id)}.swc"))
                        except Exception:
                            pass

            rows = []
            for nrn in neurons:
                nodes = nrn.nodes
                if nodes.empty:
                    continue
                parent_set = set(nodes["parent_id"].dropna().astype(int).tolist())
                internal = nodes[nodes["node_id"].isin(parent_set) & (nodes["parent_id"] >= 0)]
                if len(internal) < min_nodes:
                    internal = nodes
                pts_nm = internal[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
                sub = pd.DataFrame({
                    "skeleton_id": int(nrn.id),
                    "node_id": internal["node_id"].astype(int).to_numpy(),
                    "x": internal["x"].to_numpy(dtype=np.float64),
                    "y": internal["y"].to_numpy(dtype=np.float64),
                    "z": internal["z"].to_numpy(dtype=np.float64),
                    "x_nm": pts_nm[:, 0],
                    "y_nm": pts_nm[:, 1],
                    "z_nm": pts_nm[:, 2],
                })
                rows.append(sub)

            if not rows:
                raise RuntimeError("Loaded CATMAID skeletons contained 0 usable nodes.")

            all_df = pd.concat(rows, ignore_index=True)
            on_progress(f"Mapping {len(all_df)} CATMAID internal nodes to CAVE roots...")

            mapped = retry_call(
                lambda: map_nodes_to_cave(
                    all_df,
                    cave_client,
                    catmaid_resolution_nm=cm_res,
                    cave_resolution_nm=cave_res,
                    catmaid_coord_unit=unit,
                ),
                stage="CAVE map_nodes_to_cave",
                progress_callback=on_progress,
            )
            mapped = mapped[mapped["root_id"].notna() & (mapped["root_id"].astype(np.int64) > 0)].copy()
            mapped["root_id"] = mapped["root_id"].astype(np.int64)
            mapped["skeleton_id"] = mapped["skeleton_id"].astype(np.int64)

            # Detect false merges (1 CAVE root -> >1 CATMAID skeleton)
            splits: list[dict] = []
            for rid, grp in mapped.groupby("root_id"):
                skid_counts = grp.groupby("skeleton_id").size().sort_values(ascending=False)
                valid_skids = [int(s) for s, cnt in skid_counts.items() if int(cnt) >= min_nodes]
                if len(valid_skids) < 2:
                    continue
                src_skid = valid_skids[0]
                sink_skids = valid_skids[1:]
                src_pts = grp[grp["skeleton_id"] == src_skid][["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=float).tolist()
                sink_pts = grp[grp["skeleton_id"].isin(sink_skids)][["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=float).tolist()
                splits.append({
                    "approve": True,
                    "target_root_id": str(int(rid)),
                    "source_skid": src_skid,
                    "sink_skids": ",".join(map(str, sink_skids)),
                    "source_nodes": int(len(src_pts)),
                    "sink_nodes": int(len(sink_pts)),
                    "source_coords_nm": src_pts[:60],
                    "sink_coords_nm": sink_pts[:60],
                })

            # Detect false splits (1 CATMAID skeleton -> >1 CAVE root)
            merges: list[dict] = []
            clean_count = 0
            for skid, grp in mapped.groupby("skeleton_id"):
                root_counts = grp.groupby("root_id").size().sort_values(ascending=False)
                valid_roots = [(int(r), int(cnt)) for r, cnt in root_counts.items() if int(cnt) >= min_nodes]
                if len(valid_roots) == 1:
                    clean_count += 1
                elif len(valid_roots) > 1:
                    prim_root, prim_cnt = valid_roots[0]
                    prim_pt = grp[grp["root_id"] == prim_root][["x_nm", "y_nm", "z_nm"]].iloc[0].to_numpy(dtype=float).tolist()
                    for frag_root, frag_cnt in valid_roots[1:]:
                        frag_pt = grp[grp["root_id"] == frag_root][["x_nm", "y_nm", "z_nm"]].iloc[0].to_numpy(dtype=float).tolist()
                        merges.append({
                            "approve": True,
                            "skeleton_id": int(skid),
                            "primary_root_id": str(prim_root),
                            "fragment_root_id": str(frag_root),
                            "primary_nodes": prim_cnt,
                            "fragment_nodes": frag_cnt,
                            "primary_coord_nm": prim_pt,
                            "fragment_coord_nm": frag_pt,
                        })

            summary = {
                "total_skeletons": int(len(neurons)),
                "clean_1to1_skeletons": clean_count,
                "planned_splits": len(splits),
                "planned_merges": len(merges),
            }
            with self._lock:
                self._splits = splits
                self._merges = merges
                self._summary = summary
                self._status = "ready"
                self._progress = ""
                self._message = (
                    f"Dry-run ready: {len(neurons)} skeleton(s), {len(splits)} planned split(s), {len(merges)} planned merge(s)."
                )
        except Exception as exc:
            with self._lock:
                self._status = "failed"
                self._progress = ""
                self._message = f"Plan failed: {exc}"

    def preview_single_split(self, d: dict) -> dict:
        proj = self._plan_project or self._active_project
        if not proj:
            raise ValueError("No active project selected.")
        cfg = self._get_project_cfg(proj)
        cave_client = self._build_cave_client(cfg)
        cave_res = np.asarray(cfg["cave_resolution_nm"], dtype=np.float64)

        rid = int(d["target_root_id"])
        src_vox = np.round(np.asarray(d["source_coords_nm"], dtype=np.float64) / cave_res).astype(np.int64)
        sink_vox = np.round(np.asarray(d["sink_coords_nm"], dtype=np.float64) / cave_res).astype(np.int64)

        res = cave_client.chunkedgraph.preview_split(
            root_id=rid,
            source_points=src_vox,
            sink_points=sink_vox,
            voxel_resolution=cave_res.tolist(),
        )
        supervoxels = res.get("supervoxels") or []
        valid = bool(res.get("illegal_split") is False or len(supervoxels) >= 2)
        center_vox = np.vstack([src_vox[:1], sink_vox[:1]]).mean(axis=0).astype(int).tolist()

        ngl_url = ""
        try:
            from nglui import statebuilder

            info = cave_client.info.get_datastack_info()
            img = statebuilder.ImageLayerConfig(info["aligned_volume"]["image_source"], name="EM")
            seg = statebuilder.SegmentationLayerConfig(
                info["segmentation_source"],
                name="Segmentation",
                selected_ids=[rid],
            )
            sb = statebuilder.StateBuilder([img, seg], resolution=cave_res.tolist())
            ngl_url = sb.render_state(return_as="url", ngl_url=info.get("viewer_site"))
        except Exception:
            viewer = cave_client.info.viewer_site() or "https://neuroglancer-demo.appspot.com"
            state_obj = {
                "position": center_vox,
                "layers": [
                    {
                        "type": "segmentation",
                        "source": cave_client.info.segmentation_source(),
                        "segments": [str(rid)],
                        "name": "segmentation",
                    }
                ],
            }
            ngl_url = f"{viewer.rstrip('/')}/#!{urllib.parse.quote(json.dumps(state_obj))}"

        return {
            "split_valid": valid,
            "predicted_component_count": max(2, len(supervoxels)) if valid else 1,
            "neuroglancer_url": ngl_url,
        }

    def start_exec(self, d: dict) -> dict:
        proj = self._plan_project or self._active_project
        if not proj:
            raise ValueError("No active project selected.")
        cfg = self._get_project_cfg(proj)
        splits = [x for x in (d.get("splits") or []) if x.get("approve")]
        merges = [x for x in (d.get("merges") or []) if x.get("approve")]
        if not splits and not merges:
            raise ValueError("No splits or merges are checked for execution.")

        with self._lock:
            if self._status in ("loading", "executing"):
                raise RuntimeError("A CATMAID -> CAVE job is already running.")
            self._status = "executing"
            self._progress = f"Executing {len(splits)} split(s) and {len(merges)} merge(s) in CAVE..."
            self._audit = []

        t = threading.Thread(
            target=self._run_execute_worker,
            args=(proj, cfg, splits, merges),
            daemon=True,
        )
        t.start()
        return {"ok": True}

    def _run_execute_worker(self, proj: str, cfg: dict, splits: list[dict], merges: list[dict]) -> None:
        audit: list[dict] = []
        try:
            cave_client = self._build_cave_client(cfg)
            cave_res = np.asarray(cfg["cave_resolution_nm"], dtype=np.float64)

            for idx, sp in enumerate(splits, start=1):
                rid = int(sp["target_root_id"])
                with self._lock:
                    self._progress = f"Executing split {idx}/{len(splits)} on CAVE root {rid}..."
                try:
                    latest = cave_client.chunkedgraph.get_latest_roots(rid)
                    cur_rid = int(latest[0]) if len(latest) else rid
                    src_vox = np.round(np.asarray(sp["source_coords_nm"], dtype=np.float64) / cave_res).astype(np.int64)
                    sink_vox = np.round(np.asarray(sp["sink_coords_nm"], dtype=np.float64) / cave_res).astype(np.int64)
                    res = cave_client.chunkedgraph.do_split(
                        root_id=cur_rid,
                        source_points=src_vox,
                        sink_points=sink_vox,
                        voxel_resolution=cave_res.tolist(),
                    )
                    op_id = str(res.get("operation_id", "")) if isinstance(res, dict) else ""
                    new_roots = res.get("new_root_ids", []) if isinstance(res, dict) else []
                    new_str = ",".join(map(str, new_roots))
                    audit.append({
                        "action": "SPLIT",
                        "status": "SUCCESS",
                        "skeleton_id": sp.get("source_skid", ""),
                        "old_root_id": str(cur_rid),
                        "new_root_ids": new_str,
                        "operation_id": op_id,
                        "error": "",
                    })
                    self._record_change(
                        system="CAVE",
                        project=proj,
                        target=f"SKID {sp.get('source_skid')} vs {sp.get('sink_skids')}",
                        action="SPLIT",
                        before_ids=str(cur_rid),
                        after_ids=new_str,
                        operation_id=op_id,
                        summary=f"Split false merge on root {cur_rid}",
                    )
                except Exception as exc:
                    audit.append({
                        "action": "SPLIT",
                        "status": "FAILED",
                        "skeleton_id": sp.get("source_skid", ""),
                        "old_root_id": str(rid),
                        "new_root_ids": "",
                        "operation_id": "",
                        "error": str(exc),
                    })

            for idx, mg in enumerate(merges, start=1):
                skid = int(mg["skeleton_id"])
                r1 = int(mg["primary_root_id"])
                r2 = int(mg["fragment_root_id"])
                with self._lock:
                    self._progress = f"Executing merge {idx}/{len(merges)} for SKID {skid}..."
                try:
                    lat1 = cave_client.chunkedgraph.get_latest_roots(r1)
                    lat2 = cave_client.chunkedgraph.get_latest_roots(r2)
                    cur_r1 = int(lat1[0]) if len(lat1) else r1
                    cur_r2 = int(lat2[0]) if len(lat2) else r2
                    if cur_r1 == cur_r2:
                        audit.append({
                            "action": "MERGE",
                            "status": "SKIPPED (Already merged)",
                            "skeleton_id": skid,
                            "old_root_id": f"{cur_r1},{cur_r2}",
                            "new_root_ids": str(cur_r1),
                            "operation_id": "",
                            "error": "",
                        })
                        continue
                    p1_vox = np.round(np.asarray(mg["primary_coord_nm"], dtype=np.float64) / cave_res).astype(np.int64)
                    p2_vox = np.round(np.asarray(mg["fragment_coord_nm"], dtype=np.float64) / cave_res).astype(np.int64)
                    res = cave_client.chunkedgraph.do_merge(
                        supervoxels=None,
                        coords=[p1_vox.tolist(), p2_vox.tolist()],
                        voxel_resolution=cave_res.tolist(),
                    )
                    op_id = str(res.get("operation_id", "")) if isinstance(res, dict) else ""
                    new_roots = res.get("new_root_ids", []) if isinstance(res, dict) else []
                    new_str = ",".join(map(str, new_roots))
                    audit.append({
                        "action": "MERGE",
                        "status": "SUCCESS",
                        "skeleton_id": skid,
                        "old_root_id": f"{cur_r1},{cur_r2}",
                        "new_root_ids": new_str,
                        "operation_id": op_id,
                        "error": "",
                    })
                    self._record_change(
                        system="CAVE",
                        project=proj,
                        target=f"SKID {skid}",
                        action="MERGE",
                        before_ids=f"{cur_r1},{cur_r2}",
                        after_ids=new_str,
                        operation_id=op_id,
                        summary=f"Merged CAVE fragments for SKID {skid}",
                    )
                except Exception as exc:
                    audit.append({
                        "action": "MERGE",
                        "status": "FAILED",
                        "skeleton_id": skid,
                        "old_root_id": f"{r1},{r2}",
                        "new_root_ids": "",
                        "operation_id": "",
                        "error": str(exc),
                    })

            with self._lock:
                self._audit = audit
                self._status = "done"
                self._progress = ""
                self._message = f"Completed execution: {len(audit)} operation(s) processed."
        except Exception as exc:
            with self._lock:
                self._audit = audit
                self._status = "failed"
                self._progress = ""
                self._message = f"Execution failed: {exc}"

    # -------------------------------------------------------------------------
    # 2. CAVE -> CATMAID Export Plan & Upload
    # -------------------------------------------------------------------------

    def start_export_plan(self, d: dict) -> dict:
        proj = str(d.get("project_name") or "").strip()
        cfg = self._get_project_cfg(proj)
        root_ids = _parse_int_list(d.get("cave_roots"))
        if not root_ids:
            raise ValueError("Enter at least one valid numeric CAVE Root ID.")

        base_neuron_name = str(d.get("base_neuron_name") or "").strip()
        skeletor_method = str(d.get("skeletor_method") or "wavefront").strip().lower()
        min_twig_nm = float(d.get("min_twig_nm") if d.get("min_twig_nm") is not None else 500.0)
        resample_to_nm = float(d.get("resample_to_nm") if d.get("resample_to_nm") is not None else 160.0)
        search_annotations = str(d.get("search_annotations") or "").strip()
        cut_buffer_nm = float(d.get("cut_buffer_nm") if d.get("cut_buffer_nm") is not None else 400.0)
        slice_step = max(1, int(d.get("slice_step") if d.get("slice_step") is not None else 1))

        with self._lock:
            if self._export_status in ("loading", "executing"):
                raise RuntimeError("A CAVE -> CATMAID export job is already running.")
            self._export_status = "loading"
            self._export_progress = f"Starting export dry-run for {len(root_ids)} CAVE root(s)..."
            self._export_message = ""
            self._export_pct = 1.0
            self._export_started_at = time.time()
            self._export_elapsed_sec = None
            self._export_plan = []
            self._export_bundle = None
            self._export_audit = []
            self._export_project = proj
            self._active_project = proj

        t = threading.Thread(
            target=self._run_export_plan_worker,
            args=(
                proj,
                cfg,
                root_ids,
                base_neuron_name,
                search_annotations,
                cut_buffer_nm,
                slice_step,
                skeletor_method,
                min_twig_nm,
                resample_to_nm,
            ),
            daemon=True,
        )
        t.start()
        return {"ok": True}

    def _run_export_plan_worker(
        self,
        proj: str,
        cfg: dict,
        root_ids: list[int],
        base_neuron_name: str,
        search_annotations: str,
        cut_buffer_nm: float,
        slice_step: int,
        skeletor_method: str,
        min_twig_nm: float,
        resample_to_nm: float,
    ) -> None:
        def on_progress(msg: str, pct: float | None = None) -> None:
            with self._lock:
                self._export_progress = str(msg)
                if pct is not None:
                    self._export_pct = float(pct)

        try:
            cave_client = self._build_cave_client(cfg)
            rm = self._build_catmaid_instance(cfg)
            cave_res = tuple(float(x) for x in cfg["cave_resolution_nm"])
            cm_res = tuple(float(x) for x in cfg["catmaid_resolution_nm"])
            cm_unit = cfg.get("catmaid_coord_unit", "nm")

            plan_sig = inspect.signature(build_cave_export_plan)
            kw: dict[str, Any] = {
                "cave_root_ids": root_ids,
                "cave_client": cave_client,
                "rm": rm,
                "cave_resolution_nm": cave_res,
                "catmaid_resolution_nm": cm_res,
                "catmaid_coord_unit": cm_unit,
                "base_neuron_name": base_neuron_name,
                "search_annotations": search_annotations,
                "cut_buffer_nm": cut_buffer_nm,
                "slice_step": slice_step,
                "skeletor_method": skeletor_method,
                "resample_to_nm": resample_to_nm,
                "progress_callback": on_progress,
            }
            if "min_twig_nm" in plan_sig.parameters:
                kw["min_twig_nm"] = min_twig_nm

            plan_rows, bundle = build_cave_export_plan(**kw)

            with self._lock:
                elapsed = round(max(0.0, time.time() - (self._export_started_at or time.time())), 2)
                self._export_plan = plan_rows
                self._export_bundle = bundle
                self._export_status = "ready"
                self._export_pct = 100.0
                self._export_elapsed_sec = elapsed
                self._export_progress = ""
                self._export_message = f"Export plan ready for {len(plan_rows)} CAVE segment(s)."
        except Exception as exc:
            with self._lock:
                elapsed = round(max(0.0, time.time() - (self._export_started_at or time.time())), 2)
                self._export_status = "failed"
                self._export_elapsed_sec = elapsed
                self._export_progress = ""
                self._export_message = f"Export plan failed after {elapsed:.2f} s: {exc}"

    def start_export_exec(self, d: dict) -> dict:
        proj = self._export_project or self._active_project
        if not proj:
            raise ValueError("No active project selected.")
        cfg = self._get_project_cfg(proj)
        plan_rows = d.get("plan") or []
        approved = [r for r in plan_rows if r.get("approve")]
        if not approved:
            raise ValueError("Approve at least one CAVE segment row before exporting.")

        target_annotations = str(d.get("target_annotations") or "").strip()
        auto_stitch = bool(d.get("auto_stitch", False))

        with self._lock:
            if self._export_status in ("loading", "executing"):
                raise RuntimeError("A CAVE -> CATMAID export job is already running.")
            if not self._export_bundle:
                raise RuntimeError("Export bundle expired or missing. Rebuild the dry-run plan first.")
            bundle = self._export_bundle
            # Persist the user's current checkbox and name edits so state polls never reset them
            self._export_plan = [dict(r) for r in plan_rows]
            self._export_status = "executing"
            self._export_progress = f"Uploading {len(approved)} approved CAVE neuron(s) to CATMAID..."
            self._export_message = ""
            self._export_pct = 2.0
            self._export_started_at = time.time()
            self._export_elapsed_sec = None
            self._export_audit = []

        t = threading.Thread(
            target=self._run_export_execute_worker,
            args=(proj, cfg, plan_rows, bundle, target_annotations, auto_stitch),
            daemon=True,
        )
        t.start()
        return {"ok": True}

    def _run_export_execute_worker(
        self,
        proj: str,
        cfg: dict,
        plan_rows: list[dict],
        bundle: dict,
        target_annotations: str,
        auto_stitch: bool,
    ) -> None:
        def on_progress(msg: str, pct: float | None = None) -> None:
            with self._lock:
                self._export_progress = str(msg)
                if pct is not None:
                    self._export_pct = float(pct)

        try:
            rm = self._build_catmaid_instance(cfg)
            audit = execute_cave_export_plan(
                plan_rows=plan_rows,
                export_bundle=bundle,
                rm=rm,
                target_annotations=target_annotations,
                auto_stitch=auto_stitch,
                progress_callback=on_progress,
            )

            succeeded_roots: set[str] = set()
            for item in audit:
                if item.get("status") == "SUCCESS":
                    succeeded_roots.add(str(item.get("cave_root_id", "")))
                    self._record_change(
                        system="CATMAID",
                        project=proj,
                        target=f"CAVE {item.get('cave_root_id')}",
                        action="EXPORT_SKELETON",
                        before_ids=str(item.get("cave_root_id", "")),
                        after_ids=str(item.get("resulting_skids", "")),
                        operation_id="",
                        summary=(
                            f"Exported {item.get('fragment_names')} ({item.get('nodes_added')} nodes)"
                            + (f" [{item.get('annotations')}]" if item.get("annotations") else "")
                        ),
                    )

            with self._lock:
                elapsed = round(max(0.0, time.time() - (self._export_started_at or time.time())), 2)
                # Keep unchecked rows unchecked; mark newly exported rows as exported and unchecked
                for row in self._export_plan:
                    if str(row.get("cave_root_id", "")) in succeeded_roots:
                        row["approve"] = False
                        row["exported"] = True
                self._export_audit = audit
                self._export_status = "done"
                self._export_pct = 100.0
                self._export_elapsed_sec = elapsed
                self._export_progress = ""
                ok_cnt = sum(1 for x in audit if x.get("status") == "SUCCESS")
                self._export_message = f"Export complete: {ok_cnt}/{len(audit)} segment(s) uploaded to CATMAID."
        except Exception as exc:
            with self._lock:
                elapsed = round(max(0.0, time.time() - (self._export_started_at or time.time())), 2)
                self._export_status = "failed"
                self._export_elapsed_sec = elapsed
                self._export_progress = ""
                self._export_message = f"Export failed after {elapsed:.2f} s: {exc}"


_DEFAULT_SERVICE: CatmaidService | None = None


def init_catmaid_service(app_config: dict | None = None, base_dir: str | Path | None = None) -> CatmaidService:
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = CatmaidService(app_config=app_config, base_dir=base_dir)
    return _DEFAULT_SERVICE


def _svc() -> CatmaidService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = CatmaidService()
    return _DEFAULT_SERVICE


def _post_wrap(fn: Callable[[dict], dict], d: dict | None = None, **kwargs) -> tuple[int, dict]:
    payload = d if d is not None else kwargs
    try:
        res = fn(payload)
        return 200, res
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except RuntimeError as exc:
        return 409, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


# --- GET route functions (return raw dict / str) ---

def public_state() -> dict:
    return _svc().public_state()


get_state = public_state
state = public_state


def changelog_csv() -> str:
    return _svc().changelog_csv()


get_changelog_csv = changelog_csv
export_changelog_csv = changelog_csv


# --- POST route functions (return (http_code, payload_dict)) ---

def probe_resolutions(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().probe_resolutions, d, **kwargs)


probe = probe_resolutions
verify_connection = probe_resolutions


def save_project(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().save_project, d, **kwargs)


save_project_profile = save_project
upsert_project = save_project


def select_project(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().select_project, d, **kwargs)


set_project = select_project
set_active_project = select_project
activate_project = select_project


def delete_project(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().delete_project, d, **kwargs)


remove_project = delete_project


def start_plan(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().start_plan, d, **kwargs)


start_dry_run = start_plan
build_plan = start_plan
plan = start_plan
run_plan = start_plan


def preview_single_split(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().preview_single_split, d, **kwargs)


preview_split = preview_single_split


def start_exec(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().start_exec, d, **kwargs)


start_execute = start_exec
start_execution = start_exec
execute_plan = start_exec
execute = start_exec
run_execute = start_exec


def start_export_plan(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().start_export_plan, d, **kwargs)


build_export_plan = start_export_plan
export_plan = start_export_plan
plan_export = start_export_plan


def start_export_exec(d: dict | None = None, **kwargs) -> tuple[int, dict]:
    return _post_wrap(_svc().start_export_exec, d, **kwargs)


start_export_execute = start_export_exec
execute_export = start_export_exec
execute_export_plan = start_export_exec
export_execute = start_export_exec


init = init_catmaid_service
configure = init_catmaid_service
setup = init_catmaid_service