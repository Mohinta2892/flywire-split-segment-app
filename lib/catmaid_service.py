# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:52:00 2026

@author: Pedro Gomez Galvez
"""

"""Tab 2: Bidirectional CATMAID <-> CAVE proofreading and skeleton export workers."""
import re, time, traceback
from pathlib import Path
from lib import store

catmaid_state = {
    "status": "idle",       # idle | loading | ready | executing | done | failed
    "message": "",
    "detail": "",
    "progress": "",
    "project": None,
    "summary": None,
    "splits": [],
    "merges": [],
    "audit": [],
    "mapped_df": None,
    "export_status": "idle",
    "export_progress": "",
    "export_message": "",
    "export_pct": 0,
    "export_started_at": None,
    "export_elapsed_sec": None,
    "export_plan": [],
    "export_bundle": None,
    "export_audit": [],
}


def _cm_update(**kw):
    with store.lock:
        catmaid_state.update(kw)


def _fmt_elapsed(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60.0:
        return f"{sec:.2f} s"
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins}m {rem:.1f}s"


def public_state():
    from catmaid_cave_sync.config import load_all_configs
    cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
    safe_projects = {}
    for name, cfg in cfg_store.get("projects", {}).items():
        safe_projects[name] = {
            "cave_datastack": cfg.get("cave_datastack", ""),
            "cave_server_address": cfg.get("cave_server_address", store.SERVER),
            "cave_resolution_nm": cfg.get("cave_resolution_nm", []),
            "catmaid_url": cfg.get("catmaid_url", ""),
            "catmaid_project_id": cfg.get("catmaid_project_id", 1),
            "catmaid_http_user": cfg.get("catmaid_http_user", ""),
            "catmaid_resolution_nm": cfg.get("catmaid_resolution_nm", []),
            "catmaid_coord_unit": cfg.get("catmaid_coord_unit", "nm"),
        }
    with store.lock:
        started_at = catmaid_state.get("export_started_at")
        running_sec = (
            round(time.monotonic() - started_at, 1)
            if started_at is not None and catmaid_state["export_status"] in ("loading", "executing")
            else None
        )
        return {
            "status": catmaid_state["status"],
            "message": catmaid_state["message"],
            "detail": catmaid_state["detail"],
            "progress": catmaid_state["progress"],
            "project": catmaid_state["project"],
            "summary": catmaid_state["summary"],
            "splits": catmaid_state["splits"],
            "merges": catmaid_state["merges"],
            "audit": catmaid_state["audit"],
            "export_status": catmaid_state["export_status"],
            "export_progress": catmaid_state["export_progress"],
            "export_message": catmaid_state["export_message"],
            "export_pct": catmaid_state["export_pct"],
            "export_running_sec": running_sec,
            "export_elapsed_sec": catmaid_state["export_elapsed_sec"],
            "export_plan": catmaid_state["export_plan"],
            "export_audit": catmaid_state["export_audit"],
            "changelog": list(reversed(store.changelog)),
            "active_project": cfg_store.get("active_project"),
            "projects": safe_projects,
            "default_server": store.SERVER,
            "sample_datastacks": store.SAMPLES,
        }


def _init_cave_client(cfg):
    from caveclient import CAVEclient
    srv = cfg.get("cave_server_address") or store.SERVER
    kwargs = {"datastack_name": cfg["cave_datastack"]}
    if srv:
        kwargs["server_address"] = srv
    return CAVEclient(**kwargs)


def probe_resolutions(d):
    """Verify CATMAID & CAVE connections, query CAVE resolution, and check CATMAID stack info."""
    try:
        from caveclient import CAVEclient
        from catmaid_cave_sync.config import (
            load_all_configs,
            normalize_catmaid_url,
            query_cave_resolution,
            probe_catmaid_connection,
        )

        name = (d.get("project_name") or "").strip()
        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        existing = cfg_store.get("projects", {}).get(name, {})

        ds = (d.get("cave_datastack") or "").strip()
        cave_srv = (d.get("cave_server_address") or store.SERVER).strip()
        cm_url = normalize_catmaid_url(d.get("catmaid_url") or "")
        cm_pid = int(d.get("catmaid_project_id") or 1)
        cm_tok = (d.get("catmaid_token") or "").strip() or existing.get("catmaid_token", "")
        http_user = (d.get("catmaid_http_user") or "").strip() or existing.get("catmaid_http_user", "")
        http_pass = (d.get("catmaid_http_password") or "").strip() or existing.get("catmaid_http_password", "")

        if not (ds and cm_url and cm_tok):
            return 400, {"error": "Select a sample/datastack and enter your CATMAID URL and CATMAID API token first."}

        stacks = probe_catmaid_connection(
            server_url=cm_url,
            project_id=cm_pid,
            api_token=cm_tok,
            http_user=http_user,
            http_password=http_pass,
        )

        kwargs = {"datastack_name": ds}
        if cave_srv:
            kwargs["server_address"] = cave_srv
        cave_client = CAVEclient(**kwargs)
        cave_res = query_cave_resolution(cave_client)

        return 200, {"cave_resolution_nm": cave_res, "catmaid_stacks": stacks}
    except Exception as exc:
        return 400, {"error": store.brief(exc)}


def save_project(d):
    """Save a project profile locally to .cave_catmaid_config.json."""
    try:
        from catmaid_cave_sync.config import load_all_configs, normalize_catmaid_url, save_project_config

        name = (d.get("project_name") or "").strip()
        if not name:
            return 400, {"error": "Select or enter a project/sample name first (e.g., Parker-Goog)."}

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        existing = cfg_store.get("projects", {}).get(name, {})

        cave_res = [float(x) for x in d.get("cave_resolution_nm", [])]
        cm_res = [float(x) for x in d.get("catmaid_resolution_nm", [])]
        if len(cave_res) != 3 or len(cm_res) != 3 or any(v <= 0 for v in cave_res + cm_res):
            return 400, {"error": "Both CAVE and CATMAID resolutions must have 3 positive numbers (x,y,z nm)."}

        cm_tok = (d.get("catmaid_token") or "").strip() or existing.get("catmaid_token", "")
        if not cm_tok:
            return 400, {"error": "CATMAID API token is required."}

        http_user = (d.get("catmaid_http_user") or "").strip() or existing.get("catmaid_http_user", "")
        http_pass = (d.get("catmaid_http_password") or "").strip() or existing.get("catmaid_http_password", "")

        ds = (d.get("cave_datastack") or "").strip() or store.SAMPLES.get(name, "")
        if not ds:
            return 400, {"error": "CAVE datastack name is required."}

        cfg = {
            "cave_datastack": ds,
            "cave_server_address": (d.get("cave_server_address") or store.SERVER).strip(),
            "cave_resolution_nm": cave_res,
            "catmaid_url": normalize_catmaid_url(d.get("catmaid_url") or ""),
            "catmaid_project_id": int(d.get("catmaid_project_id") or 1),
            "catmaid_token": cm_tok,
            "catmaid_http_user": http_user,
            "catmaid_http_password": http_pass,
            "catmaid_resolution_nm": cm_res,
            "catmaid_coord_unit": d.get("catmaid_coord_unit") or "nm",
        }
        save_project_config(name, cfg, store.CATMAID_CONFIG_PATH)
        return 200, {"ok": True}
    except Exception as exc:
        return 400, {"error": store.brief(exc)}


def _run_plan_worker(project_name, mode, query_val, min_nodes, force_refresh):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.catmaid_loader import fetch_and_cache_skeletons, extract_non_end_nodes
        from catmaid_cave_sync.spatial_mapper import map_nodes_to_cave
        from catmaid_cave_sync.planner import build_proofreading_plan

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        cfg = cfg_store["projects"][project_name]
        cave_client, _ = init_clients_from_project(cfg)

        skids_or_ann = (
            query_val.strip()
            if mode == "annotation"
            else [int(x.strip()) for x in query_val.split(",") if x.strip()]
        )
        cache_dir = Path(store.ROOT_DIR) / "data" / "catmaid_skeletons" / project_name

        _cm_update(progress="Downloading / loading local CATMAID SWC skeletons...")
        neurons = fetch_and_cache_skeletons(skids_or_ann, cache_dir=cache_dir, force_refresh=force_refresh)
        if len(neurons) == 0:
            _cm_update(status="failed", message="No CATMAID skeletons matched your query.", progress="")
            return

        _cm_update(progress=f"Filtering non-end nodes across {len(neurons)} skeletons & mapping CAVE roots...")
        internal_nodes = extract_non_end_nodes(neurons)
        mapped_df = map_nodes_to_cave(
            internal_nodes,
            cave_client,
            catmaid_resolution_nm=cfg["catmaid_resolution_nm"],
            cave_resolution_nm=cfg["cave_resolution_nm"],
            catmaid_coord_unit=cfg.get("catmaid_coord_unit", "nm"),
        )
        summary, splits_df, merges_df = build_proofreading_plan(mapped_df, min_support_nodes=min_nodes)

        splits_list = splits_df.to_dict(orient="records") if not splits_df.empty else []
        for row in splits_list:
            row["target_root_id"] = str(row["target_root_id"])

        merges_list = merges_df.to_dict(orient="records") if not merges_df.empty else []
        for row in merges_list:
            row["primary_root_id"] = str(row["primary_root_id"])
            row["fragment_root_id"] = str(row["fragment_root_id"])

        _cm_update(
            status="ready",
            message=f"Loaded {summary.get('total_skeletons', 0)} skeletons ({summary.get('total_mapped_nodes', 0)} internal nodes mapped).",
            detail="",
            progress="",
            project=project_name,
            summary=summary,
            splits=splits_list,
            merges=merges_list,
            audit=[],
            mapped_df=mapped_df,
        )
    except Exception as exc:
        _cm_update(
            status="failed",
            message=f"Plan failed: {store.brief(exc)}",
            detail=traceback.format_exc()[:20000],
            progress="",
        )


def start_plan(d):
    with store.lock:
        if catmaid_state["status"] in ("loading", "executing"):
            return 409, {"error": "A CATMAID batch job is already running."}
        project_name = (d.get("project_name") or "").strip()
        mode = d.get("mode") or "annotation"
        query_val = (d.get("query") or "").strip()
        min_nodes = int(d.get("min_nodes") or 3)
        force_refresh = bool(d.get("force_refresh"))
        if not project_name:
            return 400, {"error": "Select or configure a sample/project profile first."}
        if not query_val:
            return 400, {"error": "Enter a CATMAID annotation name or skeleton IDs."}
        _cm_update(status="loading", message="", detail="", progress="Starting...", project=project_name)
    store.pool.submit(_run_plan_worker, project_name, mode, query_val, min_nodes, force_refresh)
    return 200, {"ok": True}


def preview_single_split(d):
    try:
        from catmaid_cave_sync.config import load_all_configs
        from catmaid_cave_sync.planner import preview_single_split as _preview_split

        with store.lock:
            mapped_df = catmaid_state.get("mapped_df")
            project_name = catmaid_state.get("project")
        if mapped_df is None or not project_name:
            return 400, {"error": "Build a proofreading plan first."}

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        cfg = cfg_store["projects"][project_name]
        cave_client = _init_cave_client(cfg)

        res = _preview_split(
            mapped_df,
            cave_client,
            root_id=int(d["target_root_id"]),
            source_skid=int(d["source_skid"]),
            sink_skids=[int(x) for x in str(d["sink_skids"]).split(",") if x.strip()],
            cave_resolution_nm=tuple(cfg["cave_resolution_nm"]),
        )
        return 200, res
    except Exception as exc:
        return 400, {"error": store.brief(exc)}


def _run_execute_worker(project_name, mapped_df, splits_rows, merges_rows):
    try:
        import pandas as pd
        from catmaid_cave_sync.config import load_all_configs
        from catmaid_cave_sync.cave_editor import execute_approved_edits

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        cfg = cfg_store["projects"][project_name]
        cave_client = _init_cave_client(cfg)

        splits_df = pd.DataFrame(splits_rows)
        if not splits_df.empty:
            splits_df["target_root_id"] = splits_df["target_root_id"].astype("int64")

        merges_df = pd.DataFrame(merges_rows)
        if not merges_df.empty:
            merges_df["primary_root_id"] = merges_df["primary_root_id"].astype("int64")
            merges_df["fragment_root_id"] = merges_df["fragment_root_id"].astype("int64")

        def _cb(step, total, msg):
            _cm_update(progress=f"[{step}/{total}] {msg}")

        out_dir = Path(store.ROOT_DIR) / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_path = out_dir / f"proofreading_audit_log_{project_name}.csv"

        audit_df = execute_approved_edits(
            mapped_df,
            splits_df,
            merges_df,
            cave_client,
            cave_resolution_nm=tuple(cfg["cave_resolution_nm"]),
            audit_log_path=audit_path,
            progress_callback=_cb,
        )
        audit_list = audit_df.fillna("").to_dict(orient="records") if not audit_df.empty else []
        for row in audit_list:
            row["old_root_id"] = str(row.get("old_root_id", ""))
            row["new_root_ids"] = str(row.get("new_root_ids", ""))
            if str(row.get("status", "")).upper() == "SUCCESS":
                act_raw = str(row.get("action", "")).upper()
                act_label = "CATMAID-Driven Split" if "SPLIT" in act_raw else "CATMAID-Driven Merge"
                store.record_change(
                    system="CAVE",
                    project=project_name,
                    target=cfg.get("cave_datastack", ""),
                    action=act_label,
                    before_ids=row["old_root_id"],
                    after_ids=row["new_root_ids"],
                    operation_id=str(row.get("operation_id", "")),
                    summary=f"Guided by CATMAID SKID {row.get('skeleton_id', '')}",
                )

        _cm_update(
            status="done",
            message=f"Batch complete ({len(audit_list)} edits processed). Audit saved to outputs/{audit_path.name}.",
            detail="",
            progress="",
            audit=audit_list,
        )
    except Exception as exc:
        _cm_update(
            status="failed",
            message=f"Batch execution failed: {store.brief(exc)}",
            detail=traceback.format_exc()[:20000],
            progress="",
        )


def start_execute(d):
    with store.lock:
        if catmaid_state["status"] in ("loading", "executing"):
            return 409, {"error": "A CATMAID batch job is already running."}
        mapped_df = catmaid_state.get("mapped_df")
        project_name = catmaid_state.get("project")
        if mapped_df is None or not project_name:
            return 400, {"error": "Build a proofreading plan first."}
        splits_rows = d.get("splits") or []
        merges_rows = d.get("merges") or []
        if not any(r.get("approve") for r in splits_rows + merges_rows):
            return 400, {"error": "Approve at least one split or merge row first."}
        _cm_update(status="executing", message="", detail="", progress="Starting CAVE edits...", splits=splits_rows, merges=merges_rows)
    store.pool.submit(_run_execute_worker, project_name, mapped_df, splits_rows, merges_rows)
    return 200, {"ok": True}


def _export_progress_cb(msg: str, pct: float | None = None):
    updates = {"export_progress": msg}
    if pct is not None:
        updates["export_pct"] = max(0, min(100, int(round(pct))))
    _cm_update(**updates)


def _run_export_plan_worker(
    project_name, root_ids, base_neuron_name, search_anns, cut_buffer_nm, slice_step, mip, resample_to_nm, parallel, t0
):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.cave_to_catmaid import build_cave_export_plan

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        cfg = cfg_store["projects"][project_name]
        cave_client, rm = init_clients_from_project(cfg)

        plan_rows, bundle = build_cave_export_plan(
            cave_root_ids=root_ids,
            cave_client=cave_client,
            rm=rm,
            cave_resolution_nm=tuple(cfg["cave_resolution_nm"]),
            catmaid_resolution_nm=tuple(cfg["catmaid_resolution_nm"]),
            catmaid_coord_unit=cfg.get("catmaid_coord_unit", "nm"),
            base_neuron_name=base_neuron_name,
            search_annotations=search_anns,
            cut_buffer_nm=cut_buffer_nm,
            slice_step=slice_step,
            mip=mip,
            resample_to_nm=resample_to_nm,
            parallel=parallel,
            progress_callback=_export_progress_cb,
        )
        elapsed = round(time.monotonic() - t0, 2)
        _cm_update(
            export_status="ready",
            export_progress="",
            export_pct=100,
            export_elapsed_sec=elapsed,
            export_message=(
                f"Prepared {len(plan_rows)} CAVE segment(s) for CATMAID export "
                f"(Kimimaro mip={mip}, {parallel} CPU worker(s)) in {_fmt_elapsed(elapsed)}."
            ),
            export_plan=plan_rows,
            export_bundle=bundle,
            export_audit=[],
            project=project_name,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 2)
        _cm_update(
            export_status="failed",
            export_progress="",
            export_elapsed_sec=elapsed,
            export_message=f"Export plan failed after {_fmt_elapsed(elapsed)}: {store.brief(exc)}",
        )


def start_export_plan(d):
    with store.lock:
        if catmaid_state["export_status"] in ("loading", "executing"):
            return 409, {"error": "An export job is already running."}
        project_name = (d.get("project_name") or "").strip()
        raw_roots = str(d.get("cave_roots") or "")
        root_ids = [int(x) for x in re.split(r"[\s,;]+", raw_roots.strip()) if x.strip().isdigit()]
        base_neuron_name = (d.get("base_neuron_name") or "").strip()
        search_anns = (d.get("search_annotations") or "").strip()
        cut_buffer_nm = float(d.get("cut_buffer_nm") if d.get("cut_buffer_nm") is not None else 400.0)
        slice_step = max(1, int(d.get("slice_step") if d.get("slice_step") is not None else 1))
        mip = max(0, int(d.get("mip") if d.get("mip") is not None else 4))
        resample_to_nm = float(d.get("resample_to_nm") if d.get("resample_to_nm") is not None else 240.0)
        parallel = max(1, min(16, int(d.get("parallel") if d.get("parallel") is not None else 8)))
        if not project_name:
            return 400, {"error": "Select or configure a sample/project profile first."}
        if not root_ids:
            return 400, {"error": "Enter one or more numeric CAVE root IDs to export."}
        t0 = time.monotonic()
        _cm_update(
            export_status="loading",
            export_progress=f"Starting Kimimaro CAVE skeletonization (mip={mip}, {parallel} CPU worker(s))...",
            export_message="",
            export_pct=2,
            export_started_at=t0,
            export_elapsed_sec=None,
            project=project_name,
        )
    store.pool.submit(
        _run_export_plan_worker,
        project_name,
        root_ids,
        base_neuron_name,
        search_anns,
        cut_buffer_nm,
        slice_step,
        mip,
        resample_to_nm,
        parallel,
        t0,
    )
    return 200, {"ok": True}


def _run_export_exec_worker(project_name, plan_rows, bundle, target_anns, auto_stitch, t0):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.cave_to_catmaid import execute_cave_export_plan

        cfg_store = load_all_configs(store.CATMAID_CONFIG_PATH)
        cfg = cfg_store["projects"][project_name]
        _, rm = init_clients_from_project(cfg)

        audit_rows = execute_cave_export_plan(
            plan_rows=plan_rows,
            export_bundle=bundle,
            rm=rm,
            target_annotations=target_anns,
            auto_stitch=auto_stitch,
            progress_callback=_export_progress_cb,
        )
        cm_target = f"{cfg.get('catmaid_url', '')} (pid={cfg.get('catmaid_project_id', '')})"
        for row in audit_rows:
            if str(row.get("status", "")).upper() == "SUCCESS":
                ann_txt = f" [annotations: {row['annotations']}]" if row.get("annotations") else ""
                det_txt = f" — {row['details']}" if row.get("details") else ""
                store.record_change(
                    system="CATMAID",
                    project=project_name,
                    target=cm_target,
                    action="Skeleton Export & Stitch",
                    before_ids=f"CAVE {row.get('cave_root_id', '')}",
                    after_ids=f"SKID {row.get('resulting_skids', '')}",
                    operation_id="-",
                    summary=f"{row.get('fragment_names', '')} (+{row.get('nodes_added', 0)} nodes){ann_txt}{det_txt}",
                )
        elapsed = round(time.monotonic() - t0, 2)
        _cm_update(
            export_status="done",
            export_progress="",
            export_pct=100,
            export_elapsed_sec=elapsed,
            export_message=f"Exported {len(audit_rows)} segment(s) to CATMAID in {_fmt_elapsed(elapsed)}.",
            export_audit=audit_rows,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - t0, 2)
        _cm_update(
            export_status="failed",
            export_progress="",
            export_elapsed_sec=elapsed,
            export_message=f"CATMAID export failed after {_fmt_elapsed(elapsed)}: {store.brief(exc)}",
        )


def start_export_exec(d):
    with store.lock:
        if catmaid_state["export_status"] in ("loading", "executing"):
            return 409, {"error": "An export job is already running."}
        project_name = catmaid_state.get("project")
        bundle = catmaid_state.get("export_bundle")
        plan_rows = d.get("plan") or []
        target_anns = (d.get("target_annotations") or "").strip()
        auto_stitch = bool(d.get("auto_stitch", False))
        if not project_name or not bundle:
            return 400, {"error": "Build an export dry-run plan first."}
        if not any(r.get("approve") for r in plan_rows):
            return 400, {"error": "Approve at least one CAVE segment row first."}
        t0 = time.monotonic()
        _cm_update(
            export_status="executing",
            export_progress="Uploading skeletons to CATMAID...",
            export_message="",
            export_pct=3,
            export_started_at=t0,
            export_elapsed_sec=None,
            export_plan=plan_rows,
        )
    store.pool.submit(_run_export_exec_worker, project_name, plan_rows, bundle, target_anns, auto_stitch, t0)
    return 200, {"ok": True}