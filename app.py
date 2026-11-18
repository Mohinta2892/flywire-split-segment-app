"""Web page for splitting and CATMAID-driven proofreading of segments. Makes REAL chunkedgraph edits. See README.md."""
import argparse, concurrent.futures, hmac, json, os, re, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import splitter as bs
import warnings
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(HERE, "config.json")  # samples for the menu and the CAVE server: see README.md
CATMAID_CONFIG_PATH = Path(os.path.join(HERE, ".cave_catmaid_config.json"))
SAMPLES, SERVER = {}, ""  # sample name -> datastack name, and the CAVE global server, both set by load_config()
RUNNING = {"queued", "previewing", "splitting"}
MAX_ROUNDS = 6  # 63 edits at most from one segment
TOKEN = secrets.token_urlsafe(32)
STATE_PATH = os.path.join(HERE, "outputs", "gui_jobs.json")
GUIDE_DIR = os.path.join(HERE, "end_user")  # guide.html and its screenshots, also served at /help
jobs = []  # root IDs are strings: they don't fit in a JS number
lock = threading.RLock()
pool = concurrent.futures.ThreadPoolExecutor(4)

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
    "export_plan": [],
    "export_bundle": None,
    "export_audit": [],
}


def load_config(path):
    """Read the samples and server from a JSON config. Fails at startup, with the reason, if the file is unusable."""
    global SERVER
    try:
        with open(path) as f: c = json.load(f)
    except (OSError, ValueError) as e: raise SystemExit(f"Can't read config {path}: {e}")
    s, srv = c.get("samples") if isinstance(c, dict) else None, c.get("server_address") if isinstance(c, dict) else None
    if not (isinstance(s, dict) and s and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in s.items())):
        raise SystemExit(f"Config {path}: \"samples\" must map at least one sample name to a datastack name")
    if not (isinstance(srv, str) and srv.startswith("https://")):
        raise SystemExit(f"Config {path}: \"server_address\" must be an https:// URL")
    SAMPLES.clear(); SAMPLES.update(s); SERVER = srv


def connect(sample):
    from caveclient import CAVEclient
    return CAVEclient(datastack_name=SAMPLES[sample], server_address=SERVER).chunkedgraph


def save():
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f: json.dump(jobs, f)
    os.replace(tmp, STATE_PATH)


def load():
    if not os.path.exists(STATE_PATH): return
    with open(STATE_PATH) as f: jobs[:] = json.load(f)
    for j in jobs:  # jobs did not survive the restart
        if j["status"] == "splitting":
            j.update(status="unknown", message="The page stopped during this edit, which may still have applied. Wait a few minutes, then re-check.")
        elif j["status"] in RUNNING:
            j.update(status="failed", message="The page stopped before this finished. Nothing was edited. Start it again.")


def brief(e, n=600):
    """Readable error: an HTTP error's raw response body is replaced by the server's own message. The Copy button gives the full text."""
    s = str(e); m = re.search(r"\s*for url: \S+ content:\s*b'(.*)", s, re.S)
    if m:
        inner = re.search(r'"message":\s*"((?:[^"\\]|\\.)*)"', m.group(1))
        s = s[:m.start()] + (f" ({inner.group(1)})" if inner and inner.group(1) not in s[:m.start()] else "")
    s = s if len(s) <= n else s[:n] + "… (Copy for the full text)"
    if "401" in s and "token" in s.lower(): s += " The CAVE token on the server is missing or expired: renew it (README, \"CAVE token\")."
    return s


def failure(e):
    """update() fields for an error: the full text, for the Copy button."""
    return dict(detail=str(e)[:20000])


def help_page():
    """end_user/guide.html with its screenshots inlined, so /help needs no other routes (or tokens on image URLs)."""
    import base64
    with open(os.path.join(GUIDE_DIR, "guide.html"), encoding="utf-8") as f: html = f.read()
    def inline(m):
        path = os.path.join(GUIDE_DIR, "screenshots", os.path.basename(m.group(1)))
        if not os.path.exists(path): return m.group(0)
        with open(path, "rb") as f: return 'src="data:image/png;base64,' + base64.b64encode(f.read()).decode() + '"'
    return re.sub(r'src="screenshots/([\w.-]+\.png)"', inline, html)


def update(job, **kw):
    if "message" in kw: kw.setdefault("detail", "")  # a new message replaces the old error detail
    with lock:
        job.update(kw); save()


def paths(root):
    out = os.path.join(HERE, "outputs", root); os.makedirs(out, exist_ok=True)
    return os.path.join(out, "preview_terminals.json"), os.path.join(out, "edit_log.csv")


def add(sample, root, rounds, parent, preview):
    """Record a job and queue it. A previewed job splits without asking if its preview passes."""
    with lock:
        j = dict(id=len(jobs), sample=sample, root=root, status="queued", message="", started=time.time(), rounds=rounds, parent=parent)
        jobs.append(j); save()
    if preview: pool.submit(run_preview, j)
    else: pool.submit(run_split, j, True)


def not_run(j):
    left = j.get("rounds", 1) - 1
    return f" {left} more round(s) not run on this branch." if left else ""


def run_preview(j):
    update(j, status="previewing", started=time.time())
    try:
        p = bs.preview(connect(j["sample"]), int(j["root"]), paths(j["root"])[0])
        msg = (f"Would give {p['source_l2']} / {p['sink_l2']} L2 ({p['both_l2']} on both sides, {p['neither_l2']} on neither). "
               f"Missing supervoxels {p['missing_sv']} of {p['total_sv']}, links between sides {p['links']}, {p['seconds']} s.")
        update(j, status="preview_passed" if p["passed"] else "preview_failed", message=msg)
    except Exception as e:
        update(j, status="failed", message=f"Preview failed, nothing was edited: {brief(e)}", **failure(e))
    else:
        if p["passed"]: run_split(j, False)  # a passed preview is the approval: the cut uses its exact terminals


def run_split(j, no_preview):
    update(j, status="splitting", started=time.time(), message="")
    try: cg = connect(j["sample"])  # a login or connection failure happens before any edit is sent
    except Exception as e: return update(j, status="failed", message=f"Nothing was edited, couldn't connect to CAVE: {brief(e)}", **failure(e))
    try:
        r = bs.execute(cg, int(j["root"]), *paths(j["root"]), no_preview)
    except bs.BeforeEdit as e:
        return update(j, status="failed", message=f"Nothing was edited: {brief(e)}", **failure(e))
    except Exception as e:  # the edit may have applied even though the request failed
        return update(j, status="unknown", message=f"The edit may still have applied: {brief(e)}. Wait a few minutes, then re-check.", **failure(e))
    sized = len(r["sizes"]) == len(r["new"])
    pieces = sorted(zip(r["new"], r["sizes"]), key=lambda x: -x[1]) if sized else [(x, None) for x in r["new"]]
    msg = f"Op {r['op']}, {r['seconds']} s: " + ", ".join(f"{x} ({n} L2)" if n is not None else str(x) for x, n in pieces) + "."
    if r["verify_error"]: msg += f" The edit applied, but its checks failed: {r['verify_error']}"
    clean = sized and r["separated"] is True and not r["verify_error"]  # only a verified cut is cut again
    if not clean: msg += not_run(j)
    update(j, status="not_separated" if r["separated"] is False else "split", message=msg)
    if clean and j.get("rounds", 1) > 1:
        for root, n in pieces[:2]:  # the two large pieces; fragments and too-small pieces stop here
            if n >= bs.MIN_L2: add(j["sample"], str(root), j["rounds"] - 1, j["id"], not no_preview)


def run_recheck(j):
    try:
        cg = connect(j["sample"])
        if cg.is_latest_roots([int(j["root"])])[0]:
            update(j, status="failed", message="The root hasn't changed. If the server is still working, the edit could still apply.")
        else:
            update(j, status="split", message="The edit applied. Latest roots: " + ", ".join(str(int(x)) for x in cg.get_latest_roots(int(j["root"]))) + "." + not_run(j))
    except Exception as e:
        update(j, status="unknown", message=f"Re-check failed: {brief(e)}", **failure(e))


def start(sample, root, preview, rounds=1):
    """-> (http code, error message). Every rule is enforced here, not just in the page."""
    if sample not in SAMPLES: return 400, "Unknown sample"
    if not (isinstance(root, str) and root.isascii() and root.isdigit() and 0 < int(root) < 2 ** 64):
        return 400, "Enter one segment ID, as a number"
    if not isinstance(preview, bool): return 400, "Bad request"
    if type(rounds) is not int or not 1 <= rounds <= MAX_ROUNDS: return 400, f"Rounds must be a whole number from 1 to {MAX_ROUNDS}"
    root = str(int(root))
    with lock:
        for j in jobs:
            if j["root"] == root and j["status"] in RUNNING: return 409, "This segment is already running"
            if j["root"] == root and j["status"] == "unknown": return 409, "This segment's last edit has an unknown outcome. Re-check it first."
        add(sample, root, rounds, None, preview)
    return 200, ""


def act(job_id, action):
    with lock:
        j = next((x for x in jobs if x["id"] == job_id), None)
        if not j: return 404, "Unknown job"
        if action == "split" and j["status"] == "preview_passed":
            update(j, status="queued", started=time.time()); pool.submit(run_split, j, False)
        elif action == "recheck" and j["status"] == "unknown":
            pool.submit(run_recheck, j)
        else:
            return 409, f"Can't {action} a job in this state"
    return 200, ""


# --- CATMAID-DRIVEN PROOFREADING & CAVE-TO-CATMAID EXPORT BACKEND ---

def _cm_update(**kw):
    with lock:
        catmaid_state.update(kw)


def _cm_public_state():
    from catmaid_cave_sync.config import load_all_configs
    store = load_all_configs(CATMAID_CONFIG_PATH)
    safe_projects = {}
    for name, cfg in store.get("projects", {}).items():
        safe_projects[name] = {
            "cave_datastack": cfg.get("cave_datastack", ""),
            "cave_server_address": cfg.get("cave_server_address", SERVER),
            "cave_resolution_nm": cfg.get("cave_resolution_nm", []),
            "catmaid_url": cfg.get("catmaid_url", ""),
            "catmaid_project_id": cfg.get("catmaid_project_id", 1),
            "catmaid_http_user": cfg.get("catmaid_http_user", ""),
            "catmaid_resolution_nm": cfg.get("catmaid_resolution_nm", []),
            "catmaid_coord_unit": cfg.get("catmaid_coord_unit", "nm"),
        }
    with lock:
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
            "export_plan": catmaid_state["export_plan"],
            "export_audit": catmaid_state["export_audit"],
            "active_project": store.get("active_project"),
            "projects": safe_projects,
            "default_server": SERVER,
            "sample_datastacks": SAMPLES,
        }


def _cm_init_cave_client(cfg):
    from caveclient import CAVEclient
    srv = cfg.get("cave_server_address") or SERVER
    kwargs = {"datastack_name": cfg["cave_datastack"]}
    if srv:
        kwargs["server_address"] = srv
    return CAVEclient(**kwargs)


def cm_probe_resolutions(d):
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
        store = load_all_configs(CATMAID_CONFIG_PATH)
        existing = store.get("projects", {}).get(name, {})

        ds = (d.get("cave_datastack") or "").strip()
        cave_srv = (d.get("cave_server_address") or SERVER).strip()
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
        return 400, {"error": brief(exc)}


def cm_save_project(d):
    """Save a project profile (CATMAID credentials + resolutions) locally to .cave_catmaid_config.json."""
    try:
        from catmaid_cave_sync.config import load_all_configs, normalize_catmaid_url, save_project_config

        name = (d.get("project_name") or "").strip()
        if not name:
            return 400, {"error": "Select or enter a project/sample name first (e.g., Parker-Goog)."}

        store = load_all_configs(CATMAID_CONFIG_PATH)
        existing = store.get("projects", {}).get(name, {})

        cave_res = [float(x) for x in d.get("cave_resolution_nm", [])]
        cm_res = [float(x) for x in d.get("catmaid_resolution_nm", [])]
        if len(cave_res) != 3 or len(cm_res) != 3 or any(v <= 0 for v in cave_res + cm_res):
            return 400, {"error": "Both CAVE and CATMAID resolutions must have 3 positive numbers (x,y,z nm)."}

        cm_tok = (d.get("catmaid_token") or "").strip() or existing.get("catmaid_token", "")
        if not cm_tok:
            return 400, {"error": "CATMAID API token is required."}

        http_user = (d.get("catmaid_http_user") or "").strip() or existing.get("catmaid_http_user", "")
        http_pass = (d.get("catmaid_http_password") or "").strip() or existing.get("catmaid_http_password", "")

        ds = (d.get("cave_datastack") or "").strip() or SAMPLES.get(name, "")
        if not ds:
            return 400, {"error": "CAVE datastack name is required."}

        cfg = {
            "cave_datastack": ds,
            "cave_server_address": (d.get("cave_server_address") or SERVER).strip(),
            "cave_resolution_nm": cave_res,
            "catmaid_url": normalize_catmaid_url(d.get("catmaid_url") or ""),
            "catmaid_project_id": int(d.get("catmaid_project_id") or 1),
            "catmaid_token": cm_tok,
            "catmaid_http_user": http_user,
            "catmaid_http_password": http_pass,
            "catmaid_resolution_nm": cm_res,
            "catmaid_coord_unit": d.get("catmaid_coord_unit") or "nm",
        }
        save_project_config(name, cfg, CATMAID_CONFIG_PATH)
        return 200, {"ok": True}
    except Exception as exc:
        return 400, {"error": brief(exc)}


def _run_cm_plan_worker(project_name, mode, query_val, min_nodes, force_refresh):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.catmaid_loader import fetch_and_cache_skeletons, extract_non_end_nodes
        from catmaid_cave_sync.spatial_mapper import map_nodes_to_cave
        from catmaid_cave_sync.planner import build_proofreading_plan

        store = load_all_configs(CATMAID_CONFIG_PATH)
        cfg = store["projects"][project_name]
        cave_client, _ = init_clients_from_project(cfg)

        skids_or_ann = (
            query_val.strip()
            if mode == "annotation"
            else [int(x.strip()) for x in query_val.split(",") if x.strip()]
        )
        cache_dir = Path(HERE) / "data" / "catmaid_skeletons" / project_name

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
        _cm_update(status="failed", message=f"Plan failed: {brief(exc)}", detail=str(exc)[:20000], progress="")


def cm_start_plan(d):
    with lock:
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
    pool.submit(_run_cm_plan_worker, project_name, mode, query_val, min_nodes, force_refresh)
    return 200, {"ok": True}


def cm_preview_single_split(d):
    try:
        from catmaid_cave_sync.config import load_all_configs
        from catmaid_cave_sync.planner import preview_single_split

        with lock:
            mapped_df = catmaid_state.get("mapped_df")
            project_name = catmaid_state.get("project")
        if mapped_df is None or not project_name:
            return 400, {"error": "Build a proofreading plan first."}

        store = load_all_configs(CATMAID_CONFIG_PATH)
        cfg = store["projects"][project_name]
        cave_client = _cm_init_cave_client(cfg)

        res = preview_single_split(
            mapped_df,
            cave_client,
            root_id=int(d["target_root_id"]),
            source_skid=int(d["source_skid"]),
            sink_skids=[int(x) for x in str(d["sink_skids"]).split(",") if x.strip()],
            cave_resolution_nm=tuple(cfg["cave_resolution_nm"]),
        )
        return 200, res
    except Exception as exc:
        return 400, {"error": brief(exc)}


def _run_cm_execute_worker(project_name, mapped_df, splits_rows, merges_rows):
    try:
        import pandas as pd
        from catmaid_cave_sync.config import load_all_configs
        from catmaid_cave_sync.cave_editor import execute_approved_edits

        store = load_all_configs(CATMAID_CONFIG_PATH)
        cfg = store["projects"][project_name]
        cave_client = _cm_init_cave_client(cfg)

        splits_df = pd.DataFrame(splits_rows)
        if not splits_df.empty:
            splits_df["target_root_id"] = splits_df["target_root_id"].astype("int64")

        merges_df = pd.DataFrame(merges_rows)
        if not merges_df.empty:
            merges_df["primary_root_id"] = merges_df["primary_root_id"].astype("int64")
            merges_df["fragment_root_id"] = merges_df["fragment_root_id"].astype("int64")

        def _cb(step, total, msg):
            _cm_update(progress=f"[{step}/{total}] {msg}")

        out_dir = Path(HERE) / "outputs"
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

        _cm_update(
            status="done",
            message=f"Batch complete ({len(audit_list)} edits processed). Audit saved to outputs/{audit_path.name}.",
            detail="",
            progress="",
            audit=audit_list,
        )
    except Exception as exc:
        _cm_update(status="failed", message=f"Batch execution failed: {brief(exc)}", detail=str(exc)[:20000], progress="")


def cm_start_execute(d):
    with lock:
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
    pool.submit(_run_cm_execute_worker, project_name, mapped_df, splits_rows, merges_rows)
    return 200, {"ok": True}


# --- CAVE-TO-CATMAID SKELETON EXPORT WORKERS ---

def _run_cm_export_plan_worker(project_name, root_ids, search_anns, cut_buffer_nm, slice_step):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.cave_to_catmaid import build_cave_export_plan

        store = load_all_configs(CATMAID_CONFIG_PATH)
        cfg = store["projects"][project_name]
        cave_client, rm = init_clients_from_project(cfg)

        plan_rows, bundle = build_cave_export_plan(
            cave_root_ids=root_ids,
            cave_client=cave_client,
            rm=rm,
            cave_resolution_nm=tuple(cfg["cave_resolution_nm"]),
            catmaid_resolution_nm=tuple(cfg["catmaid_resolution_nm"]),
            catmaid_coord_unit=cfg.get("catmaid_coord_unit", "nm"),
            search_annotations=search_anns,
            cut_buffer_nm=cut_buffer_nm,
            slice_step=slice_step,
            progress_callback=lambda msg: _cm_update(export_progress=msg),
        )
        _cm_update(
            export_status="ready",
            export_progress="",
            export_message=f"Prepared {len(plan_rows)} CAVE segment(s) for CATMAID export.",
            export_plan=plan_rows,
            export_bundle=bundle,
            export_audit=[],
            project=project_name,
        )
    except Exception as exc:
        _cm_update(
            export_status="failed",
            export_progress="",
            export_message=f"Export plan failed: {brief(exc)}",
        )


def cm_start_export_plan(d):
    with lock:
        if catmaid_state["export_status"] in ("loading", "executing"):
            return 409, {"error": "An export job is already running."}
        project_name = (d.get("project_name") or "").strip()
        raw_roots = str(d.get("cave_roots") or "")
        root_ids = [int(x) for x in re.split(r"[\s,;]+", raw_roots.strip()) if x.strip().isdigit()]
        search_anns = (d.get("search_annotations") or "").strip()
        cut_buffer_nm = float(d.get("cut_buffer_nm") or 400.0)
        slice_step = max(1, int(d.get("slice_step") or 4))
        if not project_name:
            return 400, {"error": "Select or configure a sample/project profile first."}
        if not root_ids:
            return 400, {"error": "Enter one or more numeric CAVE root IDs to export."}
        _cm_update(
            export_status="loading",
            export_progress="Starting CAVE skeletonization...",
            export_message="",
            project=project_name,
        )
    pool.submit(_run_cm_export_plan_worker, project_name, root_ids, search_anns, cut_buffer_nm, slice_step)
    return 200, {"ok": True}


def _run_cm_export_exec_worker(project_name, plan_rows, bundle, target_anns, auto_stitch):
    try:
        from catmaid_cave_sync.config import load_all_configs, init_clients_from_project
        from catmaid_cave_sync.cave_to_catmaid import execute_cave_export_plan

        store = load_all_configs(CATMAID_CONFIG_PATH)
        cfg = store["projects"][project_name]
        _, rm = init_clients_from_project(cfg)

        audit_rows = execute_cave_export_plan(
            plan_rows=plan_rows,
            export_bundle=bundle,
            rm=rm,
            target_annotations=target_anns,
            auto_stitch=auto_stitch,
            progress_callback=lambda msg: _cm_update(export_progress=msg),
        )
        _cm_update(
            export_status="done",
            export_progress="",
            export_message=f"Exported {len(audit_rows)} segment(s) to CATMAID.",
            export_audit=audit_rows,
        )
    except Exception as exc:
        _cm_update(
            export_status="failed",
            export_progress="",
            export_message=f"CATMAID export failed: {brief(exc)}",
        )


def cm_start_export_exec(d):
    with lock:
        if catmaid_state["export_status"] in ("loading", "executing"):
            return 409, {"error": "An export job is already running."}
        project_name = catmaid_state.get("project")
        bundle = catmaid_state.get("export_bundle")
        plan_rows = d.get("plan") or []
        target_anns = (d.get("target_annotations") or "").strip()
        auto_stitch = bool(d.get("auto_stitch", True))
        if not project_name or not bundle:
            return 400, {"error": "Build an export dry-run plan first."}
        if not any(r.get("approve") for r in plan_rows):
            return 400, {"error": "Approve at least one CAVE segment row first."}
        _cm_update(
            export_status="executing",
            export_progress="Uploading skeletons to CATMAID...",
            export_message="",
            export_plan=plan_rows,
        )
    pool.submit(_run_cm_export_exec_worker, project_name, plan_rows, bundle, target_anns, auto_stitch)
    return 200, {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # default logging would print the token

    def reply(self, code, body, ctype="application/json"):
        body = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Referrer-Policy", "no-referrer"); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(body)

    def authorised(self, given):
        if given and hmac.compare_digest(given.encode(), TOKEN.encode()): return True
        self.reply(403, {"error": "Missing or wrong token"}); return False

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            if self.authorised(parse_qs(url.query).get("token", [""])[0]): self.reply(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/help":
            if self.authorised(parse_qs(url.query).get("token", [""])[0]):
                try: self.reply(200, help_page(), "text/html; charset=utf-8")
                except OSError: self.reply(404, {"error": "The guide (end_user/guide.html) is missing"})
        elif url.path == "/api/state":
            if self.authorised(self.headers.get("X-Token", "")):
                with lock: self.reply(200, {"jobs": jobs, "samples": list(SAMPLES), "max_rounds": MAX_ROUNDS, "now": time.time()})
        elif url.path == "/api/catmaid/state":
            if self.authorised(self.headers.get("X-Token", "")):
                self.reply(200, _cm_public_state())
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorised(self.headers.get("X-Token", "")): return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 2_000_000: raise ValueError
            d = json.loads(self.rfile.read(n))
        except ValueError:
            return self.reply(400, {"error": "Bad request"})
        path = urlparse(self.path).path
        if path == "/api/start":
            code, err = start(d.get("sample"), d.get("root"), d.get("preview"), d.get("rounds", 1))
            self.reply(code, {"error": err})
        elif path == "/api/act":
            code, err = act(d.get("id"), d.get("action"))
            self.reply(code, {"error": err})
        elif path == "/api/catmaid/probe":
            code, payload = cm_probe_resolutions(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/save_project":
            code, payload = cm_save_project(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/plan":
            code, payload = cm_start_plan(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/preview_split":
            code, payload = cm_preview_single_split(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/execute":
            code, payload = cm_start_execute(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/export_plan":
            code, payload = cm_start_export_plan(d)
            self.reply(code, payload)
        elif path == "/api/catmaid/export_execute":
            code, payload = cm_start_export_exec(d)
            self.reply(code, payload)
        else:
            self.reply(404, {"error": "Not found"})


PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Segment splitter & CATMAID proofreading</title><style>
*{box-sizing:border-box}body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f4f5f7;color:#1c2330}
main{max-width:1200px;margin:auto;padding:32px 24px}h1{font-size:22px;font-weight:600;margin:0}h2{font-size:15px;font-weight:600;margin:28px 0 10px;display:flex;align-items:center;gap:6px}
.sub,.hint{color:#5b6472}.sub{margin:2px 0 16px}.hint{margin:12px 0 0;font-size:13px}
.tabs{display:flex;gap:8px;border-bottom:1px solid #d5dae2;margin-bottom:20px}
.tab{background:transparent;border:0;border-bottom:2px solid transparent;border-radius:0;padding:8px 14px;height:auto;font-weight:600;color:#5b6472;display:inline-flex;align-items:center;gap:6px}
.tab:hover{background:transparent;color:#1c2330}.tab.active{color:#2b59c3;border-bottom-color:#2b59c3}
.panel{display:none}.panel.active{display:block}
.card{background:#fff;border:1px solid #e1e4e9;border-radius:10px;box-shadow:0 1px 2px #1018280d}form.card,div.pad{padding:18px 20px}.scroll{overflow-x:auto}
.fields{display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap}label{display:grid;gap:4px;font-size:12px;font-weight:600;color:#5b6472}
.lbl-row{display:inline-flex;align-items:center;gap:5px}
label.check{display:flex;align-items:center;gap:6px;font-size:14px;font-weight:400;color:inherit;height:34px}
input:not([type=checkbox]),select,textarea{font:inherit;font-weight:400;color:inherit;padding:6px 10px;border:1px solid #cdd2da;border-radius:6px;background:#fff}
input:not([type=checkbox]),select{height:34px;padding:0 10px}#root{width:220px}#rounds{width:76px}
button{font:inherit;font-weight:500;white-space:nowrap;height:30px;padding:0 12px;border:1px solid #cdd2da;border-radius:6px;background:#fff;color:inherit;cursor:pointer}button:hover{background:#f1f3f5}
.primary{height:34px;padding:0 18px;background:#2b59c3;border-color:#2b59c3;color:#fff}.primary:hover{background:#234aa3}.danger{background:#b3261e;border-color:#b3261e;color:#fff}.danger:hover{background:#94201a}
table{border-collapse:collapse;width:100%}th{font-size:12px;font-weight:600;color:#5b6472;background:#fafbfc}th,td{border-bottom:1px solid #eceef1;padding:10px 14px;text-align:left;vertical-align:top}tr:last-child td{border-bottom:0}
.id{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;white-space:nowrap}.round{display:block;font:12px system-ui,sans-serif;color:#5b6472}.msg{color:#3d4654;font-size:13px;overflow-wrap:anywhere}.time{color:#5b6472;font-size:12px;white-space:nowrap}
.s{display:inline-block;padding:1px 9px;border-radius:999px;background:#eceef1;font-size:12px;font-weight:500;white-space:nowrap}.good{background:#dcf3e3;color:#17663a}.bad{background:#fbe2e0;color:#98211a}.busy{background:#fdf0c8;color:#7a5600}
.copy{height:22px;padding:0 8px;margin-left:8px;font-size:12px;color:#5b6472;vertical-align:baseline}.msg{user-select:text}
.top{display:flex;align-items:baseline;justify-content:space-between;gap:16px}.helplink{color:#2b59c3;font-weight:500;text-decoration:none}.helplink:hover{text-decoration:underline}
#err,#cm-err,#exp-err{color:#b3261e;margin-top:10px}#err:empty,#cm-err:empty,#exp-err:empty{display:none}td:first-child{white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.metric{background:#fff;border:1px solid #e1e4e9;border-radius:8px;padding:12px 16px}.metric .v{font-size:20px;font-weight:600;margin-top:2px}
.box{border:1px solid #d5dae2;border-radius:8px;padding:14px 16px;margin-top:14px;background:#fafbfc}.hide{display:none}
.ngl{display:inline-block;margin-left:8px;color:#2b59c3;font-weight:500;text-decoration:none}.ngl:hover{text-decoration:underline}
.ok{color:#17663a;font-weight:600}
.tip{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;border:1px solid #b8c0cc;color:#5b6472;font:600 10px/1 system-ui,sans-serif;cursor:help;opacity:.35;transition:opacity .15s,border-color .15s,color .15s,background .15s;user-select:none;flex-shrink:0}
label:hover .tip,.card:hover h2 .tip,.tab:hover .tip,.tip:hover,.tip:focus{opacity:1;border-color:#2b59c3;color:#2b59c3;background:#eef3fd;outline:none}
#tooltip{position:fixed;z-index:9999;max-width:300px;padding:8px 11px;border-radius:7px;background:#1c2330;color:#fff;font:400 12px/1.4 system-ui,-apple-system,sans-serif;box-shadow:0 6px 18px #10182838;pointer-events:none;opacity:0;transition:opacity .12s}
#tooltip.show{opacity:1}
</style></head><body><main>
<div class="top"><h1>Segment splitter</h1><a id="help" class="helplink" target="_blank" rel="noopener">How to use</a></div><p class="sub">Each split or merge is a real chunkedgraph edit.</p>

<nav class="tabs" role="tablist">
  <button type="button" class="tab active" data-tab="auto">Length-Based Auto Split <span class="tip" tabindex="0" data-tip="Automatically cut a single large CAVE false-merge segment at its most distant endpoints without needing CATMAID tracings.">i</span></button>
  <button type="button" class="tab" data-tab="catmaid">CATMAID-Driven Split &amp; Merge <span class="tip" tabindex="0" data-tip="Bidirectional sync: (1) use CATMAID skeletons to split/merge CAVE segments, and (2) export proofread CAVE skeletons back into CATMAID.">i</span></button>
</nav>

<section id="panel-auto" class="panel active">
<form id="f" class="card"><div class="fields">
  <label><span class="lbl-row">Sample <span class="tip" tabindex="0" data-tip="Target CAVE volume / datastack configured in config.json.">i</span></span><select id="sample"></select></label>
  <label><span class="lbl-row">Segment <span class="tip" tabindex="0" data-tip="64-bit CAVE segment root_id of the merged object you want to split.">i</span></span><input id="root" required inputmode="numeric" autocomplete="off"></label>
  <label><span class="lbl-row">Rounds <span class="tip" tabindex="0" data-tip="Number of recursive binary split rounds (1 round = 1 cut into 2 pieces; N rounds = up to 2^N - 1 cuts).">i</span></span><input id="rounds" type="number" min="1" value="1" required></label>
  <label class="check"><input id="preview" type="checkbox" checked> Preview first <span class="tip" tabindex="0" data-tip="Simulate the min-cut first without editing CAVE. If the preview passes quality checks, the split runs automatically.">i</span></label>
  <button class="primary">Start</button>
</div>
<p id="hint" class="hint"></p><div id="err" role="alert"></div></form>
<h2>Progress <span class="tip" tabindex="0" data-tip="Live status of queued, previewing, and completed auto-split jobs.">i</span></h2><div class="card scroll"><table id="jobs"></table></div>
</section>

<section id="panel-catmaid" class="panel">
  <div class="card pad">
    <div class="fields">
      <label><span class="lbl-row">Sample / Project Profile <span class="tip" tabindex="0" data-tip="Saved CAVE + CATMAID connection profile. Choose '+ Custom Project' to configure a new dataset.">i</span></span><select id="cm-proj"></select></label>
      <button type="button" id="cm-toggle-setup">Edit CATMAID / Resolution Setup</button>
      <span class="tip" tabindex="0" data-tip="Show or hide server URLs, authentication credentials, and voxel resolution settings for the selected project.">i</span>
    </div>
    <p id="cm-proj-info" class="hint"></p>

    <div id="cm-setup-box" class="box hide">
      <div class="fields">
        <label><span class="lbl-row">Project Name <span class="tip" tabindex="0" data-tip="Local profile name (e.g., Parker-Goog) used to store credentials and cache downloaded CATMAID skeletons.">i</span></span><input id="cm-name" placeholder="e.g. Parker-Goog"></label>
        <label><span class="lbl-row">CAVE Datastack <span class="tip" tabindex="0" data-tip="CAVE segmentation datastack name from config.json.">i</span></span><input id="cm-ds" style="width:320px" placeholder="From config.json"></label>
        <label><span class="lbl-row">CAVE Server URL <span class="tip" tabindex="0" data-tip="CAVE host URL (e.g., https://global.daf-apis.com). Uses your saved local CAVE token automatically.">i</span></span><input id="cm-srv" style="width:320px" placeholder="From config.json"></label>
      </div>
      <div class="fields" style="margin-top:12px">
        <label><span class="lbl-row">CATMAID Server URL <span class="tip" tabindex="0" data-tip="Base CATMAID URL without '?pid=...' (e.g., https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem).">i</span></span><input id="cm-url" style="width:320px" placeholder="https://neurophyla.mrc-lmb.cam.ac.uk/catmaid/fibsem"></label>
        <label><span class="lbl-row">CATMAID Project ID <span class="tip" tabindex="0" data-tip="Numeric project ID ('pid=...') from your CATMAID browser address bar.">i</span></span><input id="cm-pid" type="number" min="1" value="15" style="width:105px"></label>
        <label><span class="lbl-row">CATMAID API Token <span class="tip" tabindex="0" data-tip="Your 40-character personal CATMAID API token (in CATMAID: click your username top-right -> API token).">i</span></span><input id="cm-tok" type="password" placeholder="CATMAID API token"></label>
        <label><span class="lbl-row">HTTP User (if required) <span class="tip" tabindex="0" data-tip="Front-door HTTP Basic Auth username if your server shows a browser login popup before CATMAID loads.">i</span></span><input id="cm-http-user" style="width:150px" placeholder="neurophyla user"></label>
        <label><span class="lbl-row">HTTP Password <span class="tip" tabindex="0" data-tip="Front-door HTTP Basic Auth password (Phase 1 popup password, not your CATMAID account password).">i</span></span><input id="cm-http-pass" type="password" placeholder="HTTP password"></label>
        <button type="button" id="cm-probe">1. Verify Connection &amp; Query</button>
        <span class="tip" tabindex="0" data-tip="Test both CATMAID and CAVE authentication and auto-query available image stacks and voxel resolutions.">i</span>
      </div>
      <p id="cm-probe-msg" class="hint"></p>
      <div id="cm-res-box" class="fields" style="margin-top:12px">
        <label><span class="lbl-row">CATMAID Stack <span class="tip" tabindex="0" data-tip="Image stack found in this CATMAID project; selecting one fills its resolution if exposed by the server.">i</span></span><select id="cm-stack"><option value="">Verify connection or enter below</option></select></label>
        <label><span class="lbl-row">CATMAID Resolution (x,y,z nm) <span class="tip" tabindex="0" data-tip="Physical voxel size in nanometers (e.g., 8,8,16) used to align CATMAID tracings with CAVE.">i</span></span><input id="cm-res" placeholder="e.g. 8,8,16"></label>
        <label><span class="lbl-row">CAVE Resolution (x,y,z nm) <span class="tip" tabindex="0" data-tip="Base segmentation voxel resolution in nanometers queried from CAVE.">i</span></span><input id="cm-cave-res" placeholder="Queried from CAVE"></label>
        <label><span class="lbl-row">CATMAID Coord Unit <span class="tip" tabindex="0" data-tip="Units used by CATMAID skeleton node coordinates ('nm' in almost all projects, or 'voxels').">i</span></span><select id="cm-unit"><option value="nm">nm</option><option value="voxels">voxels</option></select></label>
        <button type="button" id="cm-save-proj" class="primary">2. Save Project Profile</button>
        <span class="tip" tabindex="0" data-tip="Save credentials and resolutions locally in .cave_catmaid_config.json so you don't need to re-enter them.">i</span>
      </div>
    </div>
  </div>

  <h2>1. CATMAID &rarr; CAVE: Proofread CAVE Segments from CATMAID Skeletons <span class="tip" tabindex="0" data-tip="Use manual CATMAID skeletons as ground truth to detect and fix false merges (splits) and false splits (merges) in CAVE.">i</span></h2>
  <div class="card pad">
    <div class="fields">
      <label><span class="lbl-row">Load Skeletons By <span class="tip" tabindex="0" data-tip="Fetch CATMAID skeletons either by a shared annotation tag or by explicit skeleton IDs.">i</span></span><select id="cm-mode"><option value="annotation">CATMAID Annotation</option><option value="skids">Skeleton IDs (comma-separated)</option></select></label>
      <label><span class="lbl-row">Annotation or Skeleton IDs <span class="tip" tabindex="0" data-tip="Exact CATMAID annotation (e.g., MBON_i1) or comma-separated skeleton IDs (e.g., 12345, 67890).">i</span></span><input id="cm-query" style="width:300px" placeholder="e.g. MBON_i1 or 12345, 67890"></label>
      <label><span class="lbl-row">Min Nodes / Side <span class="tip" tabindex="0" data-tip="Minimum internal CATMAID nodes required on each side of a conflict to propose a split or merge (filters tip noise).">i</span></span><input id="cm-min-nodes" type="number" min="1" value="3" style="width:95px"></label>
      <label class="check"><input id="cm-refresh" type="checkbox"> Re-download SWCs <span class="tip" tabindex="0" data-tip="Ignore cached SWC files on disk and download fresh skeleton geometries from CATMAID.">i</span></label>
      <button type="button" id="cm-plan-btn" class="primary">Download &amp; Build Dry-Run Plan</button>
      <span class="tip" tabindex="0" data-tip="Map CATMAID internal nodes to CAVE and generate a read-only dry-run table of proposed splits and merges.">i</span>
    </div>
    <p id="cm-status" class="hint"></p>
    <div id="cm-err" role="alert"></div>
  </div>

  <div id="cm-results" class="hide">
    <div class="metrics" id="cm-metrics"></div>

    <h2>Planned Splits (CAVE roots overlapping &gt;1 CATMAID skeleton) <span class="tip" tabindex="0" data-tip="False merges in CAVE where two or more distinct CATMAID neurons occupy the same CAVE segment.">i</span></h2>
    <div class="card scroll"><table id="cm-splits-tbl"></table></div>

    <h2>Planned Merges (CATMAID skeletons spanning &gt;1 CAVE root) <span class="tip" tabindex="0" data-tip="False splits in CAVE where a single continuous CATMAID neuron crosses multiple CAVE segments.">i</span></h2>
    <div class="card scroll"><table id="cm-merges-tbl"></table></div>

    <div style="margin-top:16px;display:flex;gap:12px;align-items:center">
      <button type="button" id="cm-exec-btn" class="danger">Execute Approved Splits &amp; Merges in CAVE</button>
      <span class="tip" tabindex="0" data-tip="Apply only the checked rows in CAVE: splits execute first, root IDs refresh, then merges execute.">i</span>
      <span class="hint" style="margin:0">Executes all checked splits first, refreshes roots, then executes checked merges.</span>
    </div>

    <div id="cm-audit-wrap" class="hide">
      <h2>Execution Audit Log <span class="tip" tabindex="0" data-tip="Record of every CAVE edit executed, including operation IDs needed if you ever want to undo a change.">i</span></h2>
      <div class="card scroll"><table id="cm-audit-tbl"></table></div>
    </div>
  </div>

  <h2>2. CAVE &rarr; CATMAID: Export Neuron Skeletons to CATMAID <span class="tip" tabindex="0" data-tip="Skeletonize proofread CAVE segments, subtract already-traced manual CATMAID branches, and graft missing backbones/arbors into CATMAID.">i</span></h2>
  <div class="card pad">
    <div class="fields">
      <label><span class="lbl-row">CAVE Root IDs <span class="tip" tabindex="0" data-tip="One or more proofread CAVE segment root_ids (comma, space, or newline separated) to skeletonize and export.">i</span></span>
        <textarea id="exp-roots" rows="2" style="width:300px" placeholder="720575940612345678, 720575940687654321"></textarea>
      </label>
      <label><span class="lbl-row">CATMAID Search Annotations <span class="tip" tabindex="0" data-tip="Existing CATMAID annotations (e.g., KC right, i1_fragments) to check for overlapping manual tracings. Leave blank to search the 3D bounding box.">i</span></span>
        <input id="exp-search-ann" style="width:225px" placeholder="e.g. KC right, i1_fragments">
      </label>
      <label><span class="lbl-row">Cut Buffer (nm) <span class="tip" tabindex="0" data-tip="Exclusion radius (nm) around existing manual CATMAID nodes. CAVE nodes inside this radius are cut away so manual tracings and synapses are never duplicated.">i</span></span>
        <input id="exp-buffer" type="number" min="50" value="400" style="width:95px">
      </label>
      <label><span class="lbl-row">Z-Slice Step <span class="tip" tabindex="0" data-tip="Keep 1 node every N Z-slices along smooth neurites (CATMAID interpolates intermediate slices), while preserving every branch point, tip, and bend.">i</span></span>
        <input id="exp-slice-step" type="number" min="1" max="20" value="4" style="width:85px">
      </label>
      <label><span class="lbl-row">Target CATMAID Annotations <span class="tip" tabindex="0" data-tip="Comma-separated CATMAID annotations (e.g., KC right, KC mature) applied to all updated and newly exported skeletons.">i</span></span>
        <input id="exp-target-ann" style="width:220px" placeholder="e.g. KC right, KC mature">
      </label>
      <label class="check"><input id="exp-stitch" type="checkbox" checked> Auto-stitch into Master <span class="tip" tabindex="0" data-tip="Join exported missing subtrees (and bridged fragments) directly onto the paired Master CATMAID skeleton so manual synapses stay unified.">i</span></label>
      <button type="button" id="exp-plan-btn" class="primary">Build Export Dry-Run Plan</button>
      <span class="tip" tabindex="0" data-tip="Preview pairing, node counts after cutting/decimation, and split fragment names (neuronX_1, neuronX_2, neuronX_3...) before uploading anything to CATMAID.">i</span>
    </div>
    <p class="hint">Pairs CAVE segments with manual CATMAID nodes, cuts overlapping regions, places nodes every N Z-slices, names 3+ split fragments as neuronX_1, neuronX_2, neuronX_3..., and tags imported nodes with "cave_imported".</p>
    <p id="exp-status" class="hint"></p>
    <div id="exp-err" role="alert"></div>
  </div>

  <div id="exp-results" class="hide">
    <h2>Export Dry-Run Plan <span class="tip" tabindex="0" data-tip="Review how each CAVE segment matched existing CATMAID skeletons and which missing branches will be uploaded.">i</span></h2>
    <div class="card scroll"><table id="exp-tbl"></table></div>
    <div style="margin-top:16px;display:flex;gap:12px;align-items:center">
      <button type="button" id="exp-exec-btn" class="primary">Export Approved Skeletons to CATMAID</button>
      <span class="tip" tabindex="0" data-tip="Upload approved non-overlapping fragments to CATMAID, tag them 'cave_imported', stitch if enabled, and apply Target CATMAID Annotations.">i</span>
      <span class="hint" style="margin:0">Uploads only non-overlapping missing branches and applies Target CATMAID Annotations.</span>
    </div>
    <div id="exp-audit-wrap" class="hide">
      <h2>CATMAID Export Log <span class="tip" tabindex="0" data-tip="Summary of resulting CATMAID skeleton IDs, uploaded node counts, and applied annotations.">i</span></h2>
      <div class="card scroll"><table id="exp-audit-tbl"></table></div>
    </div>
  </div>
</section>
<div id="tooltip" role="tooltip"></div>

<script>
const T=new URLSearchParams(location.search).get('token'),$=i=>document.getElementById(i);$('help').href='help?token='+encodeURIComponent(T);
const api=async(p,b)=>{const r=await fetch(p,{method:b?'POST':'GET',headers:{'X-Token':T},body:b&&JSON.stringify(b)});const j=await r.json();
 if(!r.ok)throw new Error(j.error||r.status);return j};
const LABEL={queued:'Queued',previewing:'Previewing',splitting:'Splitting',preview_passed:'Preview passed',preview_failed:'Preview failed',split:'Split',
 not_separated:'Split, sides not separated',failed:'Failed',unknown:'Outcome unknown'};
const CLS={preview_passed:'good',split:'good',preview_failed:'bad',failed:'bad',unknown:'bad',not_separated:'bad',queued:'busy',previewing:'busy',splitting:'busy'};
const el=(t,txt,c)=>{const e=document.createElement(t);if(txt!=null)e.textContent=txt;if(c)e.className=c;return e};
const safe=f=>async()=>{try{$('err').textContent='';await f();await draw()}catch(e){$('err').textContent=e.message}};
const btn=(txt,fn,c)=>{const b=el('button',txt,c);b.onclick=safe(fn);return b};

const tipBox=$('tooltip');let pinnedTip=null;
function showTip(target){
  const txt=target&&target.dataset?target.dataset.tip:'';
  if(!txt){tipBox.classList.remove('show');return}
  tipBox.textContent=txt;tipBox.classList.add('show');
  const r=target.getBoundingClientRect(),tr=tipBox.getBoundingClientRect();
  let left=r.left+r.width/2-tr.width/2;
  left=Math.max(10,Math.min(window.innerWidth-tr.width-10,left));
  let top=r.top-tr.height-8;
  if(top<10)top=r.bottom+8;
  tipBox.style.left=left+'px';tipBox.style.top=top+'px';
}
function hideTip(){pinnedTip=null;tipBox.classList.remove('show')}
document.addEventListener('mouseover',e=>{const t=e.target.closest('.tip');if(t)showTip(t)});
document.addEventListener('mouseout',e=>{const t=e.target.closest('.tip');if(t&&t!==pinnedTip)tipBox.classList.remove('show')});
document.addEventListener('focusin',e=>{const t=e.target.closest('.tip');if(t)showTip(t)});
document.addEventListener('focusout',e=>{if(e.target.closest('.tip'))hideTip()});
document.addEventListener('click',e=>{const t=e.target.closest('.tip');if(t){e.preventDefault();e.stopPropagation();pinnedTip=t;showTip(t)}else if(pinnedTip){hideTip()}});

function copyText(text,b){const done=()=>{b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1500)};
 if(navigator.clipboard&&window.isSecureContext)return navigator.clipboard.writeText(text).then(done,()=>fallback());fallback();
 function fallback(){const a=el('textarea');a.value=text;a.style.position='fixed';a.style.opacity='0';document.body.append(a);a.select();
  try{document.execCommand('copy');done()}catch(e){prompt('Copy with Ctrl+C / Cmd+C:',text)}a.remove()}}
const edit=(r,n)=>confirm(n>1?`Up to ${2**n-1} real edits over ${n} rounds, starting with ${r}. Later rounds start without asking. Continue?`:`Real edit on ${r}. It can't be cancelled. Continue?`);
function hint(){const n=+$('rounds').value||1;$('hint').textContent=(n>1?`Up to ${2**n-1} edits and ${2**n} large pieces. `:'')+($('preview').checked
 ?(n>1?'Each cut is previewed first and made without asking if its preview passes.':'The cut is made without asking if its preview passes.')
 :(n>1?'No previews: every cut is made without asking, after one confirmation.':'No preview: the cut is made after one confirmation.'))}
function row(j,now,depth){const r=el('tr'),seg=el('td',null,'id'),st=el('td'),a=el('td'),n=j.rounds||1;seg.style.paddingLeft=14+depth*20+'px';seg.append((depth?'└ ':'')+j.root);
 if(depth+n>1)seg.append(el('span',`Round ${depth+1} of ${depth+n}`,'round'));
 st.append(el('span',LABEL[j.status]||j.status,'s '+(CLS[j.status]||'')));
 if(CLS[j.status]=='busy')st.append(el('div',Math.round(now-j.started)+' s','time'));
 if(j.status=='preview_passed')a.append(btn('Split',()=>edit(j.root,n)?api('/api/act',{id:j.id,action:'split'}):0,'danger'));
 if(j.status=='unknown')a.append(btn('Re-check',()=>api('/api/act',{id:j.id,action:'recheck'})));
 const m=el('td',null,'msg');m.append(el('span',j.message));
 if(j.message){const b=el('button','Copy','copy');b.title='Copy the full result, including any raw server error';
  b.onclick=()=>copyText(`${j.sample} ${j.root} ${LABEL[j.status]||j.status}\n${j.message}`+(j.detail?`\n\nFull error:\n${j.detail}`:''),b);m.append(b)}
 r.append(el('td',j.sample),seg,st,m,a);return r}
const selecting=()=>{const s=getSelection();return s&&!s.isCollapsed&&$('jobs').contains(s.anchorNode)};
async function draw(){const d=await api('/api/state');if(selecting())return;const s=$('sample');if(!s.options.length)for(const x of d.samples)s.append(new Option(x,x));$('rounds').max=d.max_rounds;
 const t=$('jobs'),h=el('tr');for(const x of['Sample','Segment','Status','Result',''])h.append(el('th',x));t.replaceChildren(h);
 if(!d.jobs.length){const r=el('tr'),c=el('td','Nothing yet.','msg');c.colSpan=5;r.append(c);t.append(r)}
 const kids={};for(const j of d.jobs)(kids[j.parent]??=[]).push(j);
 const walk=(j,depth)=>{t.append(row(j,d.now,depth));(kids[j.id]||[]).forEach(c=>walk(c,depth+1))};
 d.jobs.filter(j=>j.parent==null).reverse().forEach(j=>walk(j,0))}
$('f').oninput=hint;hint();
$('f').onsubmit=e=>{e.preventDefault();safe(async()=>{const root=$('root').value.trim(),preview=$('preview').checked,rounds=+$('rounds').value;if(!preview&&!edit(root,rounds))return;
 await api('/api/start',{sample:$('sample').value,root,preview,rounds});$('root').value=''})()};
draw().catch(e=>$('err').textContent=e.message);setInterval(()=>draw().catch(e=>$('err').textContent='Lost contact with the server: '+e.message),3000);

document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');$('panel-'+b.dataset.tab).classList.add('active');
});

let cmCache=null,cmSplits=[],cmMerges=[],cmRenderedKey='',cmConfiguredSet=new Set();
let expPlan=[],expRenderedKey='';
const cmSafe=f=>async()=>{try{$('cm-err').textContent='';await f()}catch(e){$('cm-err').textContent=e.message}};
const expSafe=f=>async()=>{try{$('exp-err').textContent='';await f()}catch(e){$('exp-err').textContent=e.message}};

function fillProjectForm(name,d){
  const projects=d.projects||{},samples=d.sample_datastacks||{},defaultSrv=d.default_server||'';
  const cfg=projects[name];
  $('cm-probe-msg').textContent='';$('cm-probe-msg').classList.remove('ok');
  if(!cfg){
    $('cm-name').value=(name&&name!=='+ Custom Project')?name:'';
    $('cm-ds').value=samples[name]||'';
    $('cm-srv').value=defaultSrv;
    $('cm-url').value='';$('cm-pid').value='15';$('cm-tok').value='';
    $('cm-http-user').value='';$('cm-http-pass').value='';
    $('cm-res').value='';$('cm-cave-res').value='';$('cm-unit').value='nm';
    $('cm-proj-info').textContent=`First-time setup for "${name||'new project'}": enter your CATMAID credentials below and click Verify Connection & Query.`;
    $('cm-setup-box').classList.remove('hide');
    return;
  }
  $('cm-name').value=name;$('cm-ds').value=cfg.cave_datastack||samples[name]||'';$('cm-srv').value=cfg.cave_server_address||defaultSrv;
  $('cm-url').value=cfg.catmaid_url||'';$('cm-pid').value=cfg.catmaid_project_id||15;$('cm-tok').value='';
  $('cm-http-user').value=cfg.catmaid_http_user||'';$('cm-http-pass').value='';
  $('cm-res').value=(cfg.catmaid_resolution_nm||[]).join(',');$('cm-cave-res').value=(cfg.cave_resolution_nm||[]).join(',');
  $('cm-unit').value=cfg.catmaid_coord_unit||'nm';
  $('cm-proj-info').textContent=`Loaded saved profile: ${name} | Datastack: ${$('cm-ds').value} | CATMAID res: [${$('cm-res').value}] nm | CAVE res: [${$('cm-cave-res').value}] nm`;
}

function renderCatmaidTables(d){
  const key=JSON.stringify([d.status,d.summary,d.splits.length,d.merges.length,d.audit.length]);
  if(key===cmRenderedKey)return;
  cmRenderedKey=key;
  if(!d.summary){$('cm-results').classList.add('hide');return}
  $('cm-results').classList.remove('hide');
  cmSplits=d.splits.map(x=>({...x}));cmMerges=d.merges.map(x=>({...x}));

  const m=$('cm-metrics');m.replaceChildren();
  [['Skeletons Loaded',d.summary.total_skeletons],['Clean 1-to-1',d.summary.clean_1to1_skeletons],
   ['Planned Splits',d.summary.planned_splits],['Planned Merges',d.summary.planned_merges]].forEach(([lbl,val])=>{
    const c=el('div',null,'metric');c.append(el('div',lbl,'hint'),el('div',String(val),'v'));m.append(c);
  });

  const st=$('cm-splits-tbl'),sh=el('tr');
  ['Approve','CAVE Root ID','Source Skid','Sink Skids','Source / Sink Nodes','3D Preview'].forEach(x=>sh.append(el('th',x)));
  st.replaceChildren(sh);
  if(!cmSplits.length){const r=el('tr'),c=el('td','No false merges detected (0 splits needed).','msg');c.colSpan=6;r.append(c);st.append(r)}
  cmSplits.forEach((row,idx)=>{
    const r=el('tr'),cb=el('input');cb.type='checkbox';cb.checked=!!row.approve;cb.onchange=()=>cmSplits[idx].approve=cb.checked;
    const tdCb=el('td');tdCb.append(cb);
    const tdAct=el('td'),out=el('span',null,'msg');
    const pb=el('button','Preview 3D');
    pb.onclick=cmSafe(async()=>{
      pb.disabled=true;out.textContent=' Querying CAVE...';
      try{
        const res=await api('/api/catmaid/preview_split',row);
        out.textContent=res.split_valid?` Valid (${res.predicted_component_count} pieces)`:' Invalid cut';
        const a=el('a','Open Neuroglancer','ngl');a.href=res.neuroglancer_url;a.target='_blank';a.rel='noopener';out.append(a);
      }finally{pb.disabled=false}
    });
    tdAct.append(pb,out);
    r.append(tdCb,el('td',row.target_root_id,'id'),el('td',String(row.source_skid),'id'),el('td',String(row.sink_skids),'id'),
             el('td',`${row.source_nodes} / ${row.sink_nodes}`),tdAct);
    st.append(r);
  });

  const mt=$('cm-merges-tbl'),mh=el('tr');
  ['Approve','CATMAID Skid','Primary Root ID','Fragment Root ID','Primary / Fragment Nodes'].forEach(x=>mh.append(el('th',x)));
  mt.replaceChildren(mh);
  if(!cmMerges.length){const r=el('tr'),c=el('td','No false splits detected (0 merges needed).','msg');c.colSpan=5;r.append(c);mt.append(r)}
  cmMerges.forEach((row,idx)=>{
    const r=el('tr'),cb=el('input');cb.type='checkbox';cb.checked=!!row.approve;cb.onchange=()=>cmMerges[idx].approve=cb.checked;
    const tdCb=el('td');tdCb.append(cb);
    r.append(tdCb,el('td',String(row.skeleton_id),'id'),el('td',row.primary_root_id,'id'),el('td',row.fragment_root_id,'id'),
             el('td',`${row.primary_nodes} / ${row.fragment_nodes}`));
    mt.append(r);
  });

  if(d.audit&&d.audit.length){
    $('cm-audit-wrap').classList.remove('hide');
    const at=$('cm-audit-tbl'),ah=el('tr');
    ['Action','Status','Skeleton ID','Old Root','New Root(s)','Op ID / Error'].forEach(x=>ah.append(el('th',x)));
    at.replaceChildren(ah);
    d.audit.forEach(a=>{
      const r=el('tr');
      r.append(el('td',a.action),el('td',a.status),el('td',String(a.skeleton_id),'id'),el('td',a.old_root_id,'id'),
               el('td',a.new_root_ids,'id'),el('td',a.error||String(a.operation_id||''),'msg'));
      at.append(r);
    });
  }else{$('cm-audit-wrap').classList.add('hide')}
}

function renderExportTables(d){
  const key=JSON.stringify([d.export_status,(d.export_plan||[]).length,(d.export_audit||[]).length]);
  if(key===expRenderedKey)return;
  expRenderedKey=key;
  if(!d.export_plan||!d.export_plan.length){$('exp-results').classList.add('hide');return}
  $('exp-results').classList.remove('hide');
  expPlan=d.export_plan.map(x=>({...x}));

  const et=$('exp-tbl'),eh=el('tr');
  ['Approve','CAVE Root ID','Match Type','Paired CATMAID SKID(s)','Kept / Total Nodes','Split Fragment Name(s)'].forEach(x=>eh.append(el('th',x)));
  et.replaceChildren(eh);
  expPlan.forEach((row,idx)=>{
    const r=el('tr'),cb=el('input');cb.type='checkbox';cb.checked=!!row.approve;cb.onchange=()=>expPlan[idx].approve=cb.checked;
    const tdCb=el('td');tdCb.append(cb);
    r.append(tdCb,el('td',row.cave_root_id,'id'),el('td',row.match_type),el('td',row.paired_skids,'id'),
             el('td',`${row.kept_nodes} / ${row.total_nodes}`),el('td',row.fragment_names,'msg'));
    et.append(r);
  });

  if(d.export_audit&&d.export_audit.length){
    $('exp-audit-wrap').classList.remove('hide');
    const at=$('exp-audit-tbl'),ah=el('tr');
    ['CAVE Root ID','Status','Resulting CATMAID SKID(s)','Fragment Names','Nodes Added','Annotations','Details'].forEach(x=>ah.append(el('th',x)));
    at.replaceChildren(ah);
    d.export_audit.forEach(a=>{
      const r=el('tr');
      r.append(el('td',a.cave_root_id,'id'),el('td',a.status),el('td',a.resulting_skids,'id'),
               el('td',a.fragment_names,'msg'),el('td',String(a.nodes_added)),el('td',a.annotations,'msg'),el('td',a.details,'msg'));
      at.append(r);
    });
  }else{$('exp-audit-wrap').classList.add('hide')}
}

async function drawCatmaid(){
  const d=await api('/api/catmaid/state');cmCache=d;
  const sel=$('cm-proj');
  const allNames=Array.from(new Set([...Object.keys(d.sample_datastacks||{}),...Object.keys(d.projects||{})]));
  const opts=[...allNames,'+ Custom Project'];
  if(sel.options.length!==opts.length||[...sel.options].some((o,i)=>o.value!==opts[i])){
    const prev=sel.value;sel.replaceChildren();
    opts.forEach(n=>sel.append(new Option(n,n)));
    sel.value=allNames.includes(prev)?prev:(d.active_project&&allNames.includes(d.active_project)?d.active_project:opts[0]);
    fillProjectForm(sel.value,d);
  }
  cmConfiguredSet=new Set(Object.keys(d.projects||{}));
  $('cm-status').textContent=d.progress||d.message||'';
  $('cm-plan-btn').disabled=(d.status==='loading'||d.status==='executing');
  $('cm-exec-btn').disabled=(d.status==='loading'||d.status==='executing');
  $('exp-status').textContent=d.export_progress||d.export_message||'';
  $('exp-plan-btn').disabled=(d.export_status==='loading'||d.export_status==='executing');
  $('exp-exec-btn').disabled=(d.export_status==='loading'||d.export_status==='executing');
  renderCatmaidTables(d);
  renderExportTables(d);
}

$('cm-proj').onchange=()=>{if(cmCache){$('cm-setup-box').classList.add('hide');fillProjectForm($('cm-proj').value,cmCache)}};
$('cm-toggle-setup').onclick=()=>$('cm-setup-box').classList.toggle('hide');

$('cm-probe').onclick=cmSafe(async()=>{
  const pm=$('cm-probe-msg');pm.classList.remove('ok');
  pm.textContent='Verifying CATMAID and CAVE connections...';
  const res=await api('/api/catmaid/probe',{
    project_name:$('cm-name').value,
    cave_datastack:$('cm-ds').value,cave_server_address:$('cm-srv').value,
    catmaid_url:$('cm-url').value,catmaid_project_id:+$('cm-pid').value,catmaid_token:$('cm-tok').value,
    catmaid_http_user:$('cm-http-user').value,catmaid_http_password:$('cm-http-pass').value
  });
  $('cm-cave-res').value=(res.cave_resolution_nm||[]).join(',');
  const st=$('cm-stack');st.replaceChildren();
  const stacks=res.catmaid_stacks||[];
  if(stacks.length){
    stacks.forEach(s=>{
      const rTxt=s.resolution_nm?`${s.resolution_nm.join('x')} nm`:'enter resolution manually below';
      st.append(new Option(`${s.title} [ID ${s.stack_id}] (${rTxt})`,s.resolution_nm?s.resolution_nm.join(','):''));
    });
    if(stacks[0].resolution_nm){$('cm-res').value=stacks[0].resolution_nm.join(',')}
    const names=stacks.map(s=>`${s.title} (ID ${s.stack_id})`).join(', ');
    pm.textContent=`Connected to CATMAID & CAVE! Verified stack(s): ${names}. Confirm or enter CATMAID resolution (x,y,z nm) below and click Save Project Profile.`;
  }else{
    st.append(new Option('Connected - enter resolution manually below',''));
    pm.textContent='Connected to CATMAID & CAVE! Enter CATMAID resolution (x,y,z nm) below and click Save Project Profile.';
  }
  pm.classList.add('ok');
});
$('cm-stack').onchange=()=>{if($('cm-stack').value)$('cm-res').value=$('cm-stack').value};

$('cm-save-proj').onclick=cmSafe(async()=>{
  await api('/api/catmaid/save_project',{
    project_name:$('cm-name').value,cave_datastack:$('cm-ds').value,cave_server_address:$('cm-srv').value,
    cave_resolution_nm:$('cm-cave-res').value.split(',').map(Number),
    catmaid_url:$('cm-url').value,catmaid_project_id:+$('cm-pid').value,catmaid_token:$('cm-tok').value,
    catmaid_http_user:$('cm-http-user').value,catmaid_http_password:$('cm-http-pass').value,
    catmaid_resolution_nm:$('cm-res').value.split(',').map(Number),catmaid_coord_unit:$('cm-unit').value
  });
  $('cm-setup-box').classList.add('hide');
  $('cm-status').textContent='Project profile saved locally to .cave_catmaid_config.json.';
  $('cm-proj').replaceChildren();
  await drawCatmaid();
});

$('cm-plan-btn').onclick=cmSafe(async()=>{
  const proj=$('cm-proj').value;
  if(!proj||!cmConfiguredSet.has(proj))throw new Error('Complete and save the first-time CATMAID & resolution setup above first.');
  await api('/api/catmaid/plan',{
    project_name:proj,mode:$('cm-mode').value,query:$('cm-query').value,
    min_nodes:+$('cm-min-nodes').value||3,force_refresh:$('cm-refresh').checked
  });
  await drawCatmaid();
});

$('cm-exec-btn').onclick=cmSafe(async()=>{
  const nS=cmSplits.filter(x=>x.approve).length,nM=cmMerges.filter(x=>x.approve).length;
  if(!confirm(`Execute ${nS} split(s) and ${nM} merge(s) in CAVE? Real chunkedgraph edits will be made.`))return;
  await api('/api/catmaid/execute',{splits:cmSplits,merges:cmMerges});
  await drawCatmaid();
});

$('exp-plan-btn').onclick=expSafe(async()=>{
  const proj=$('cm-proj').value;
  if(!proj||!cmConfiguredSet.has(proj))throw new Error('Complete and save the first-time CATMAID & resolution setup above first.');
  await api('/api/catmaid/export_plan',{
    project_name:proj,
    cave_roots:$('exp-roots').value,
    search_annotations:$('exp-search-ann').value,
    cut_buffer_nm:+$('exp-buffer').value||400,
    slice_step:+$('exp-slice-step').value||4
  });
  await drawCatmaid();
});

$('exp-exec-btn').onclick=expSafe(async()=>{
  const nE=expPlan.filter(x=>x.approve).length;
  if(!confirm(`Upload approved CAVE skeleton fragments for ${nE} segment(s) into CATMAID?`))return;
  await api('/api/catmaid/export_execute',{
    plan:expPlan,
    target_annotations:$('exp-target-ann').value,
    auto_stitch:$('exp-stitch').checked
  });
  await drawCatmaid();
});

drawCatmaid().catch(()=>{});setInterval(()=>drawCatmaid().catch(()=>{}),3000);
</script></main></body></html>"""


def selftest():
    """Whole flow over real HTTP against a fake chunkedgraph. Makes no CAVE calls."""
    import tempfile, urllib.error, urllib.request
    bs.selftest()
    global STATE_PATH, connect, paths
    d = tempfile.mkdtemp(); STATE_PATH = os.path.join(d, "jobs.json")
    cfg = os.path.join(d, "config.json")
    for bad in ("{", '{"samples": {}, "server_address": "https://x"}', '{"samples": {"A": "a"}, "server_address": "http://x"}'):
        with open(cfg, "w") as f: f.write(bad)
        try: load_config(cfg); raise AssertionError(f"bad config must be refused: {bad}")
        except SystemExit: pass
    with open(cfg, "w") as f: json.dump({"server_address": "https://x", "samples": {"NGC-11": "ngc11_datastack", "B": "b"}}, f)
    load_config(cfg); assert SERVER == "https://x" and SAMPLES == {"NGC-11": "ngc11_datastack", "B": "b"}

    class FakeCG:
        def is_latest_roots(self, r): return [False]
        def get_latest_roots(self, r): return [99]
    calls = []
    connect = lambda sample: (calls.append(("connect", sample)), FakeCG())[1]
    paths = lambda root: ("p", "l")
    bs.preview = lambda cg, r, p: (calls.append(("preview", r)), dict(passed=r != 3, legal=True, source_l2=1, sink_l2=1, both_l2=0, neither_l2=0, missing_sv=0, total_sv=9, links=0, seconds=0))[1]
    bs.execute = lambda cg, r, p, l, np: (calls.append(("execute", r, np)), dict(op=7, new=[r * 10, r * 10 + 1], sizes=[5, 50], separated=True, verify_error=None, seconds=0))[1]
    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler); threading.Thread(target=s.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{s.server_address[1]}"

    def req(path, body=None, token=TOKEN):
        r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={"X-Token": token})
        try:
            with urllib.request.urlopen(r) as x: return x.status, json.loads(x.read() or b"{}")
        except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

    def wait(job_id, status):
        for _ in range(200):
            st = req("/api/state")[1]["jobs"][job_id]["status"]
            if st == status: return
            time.sleep(.02)
        raise AssertionError(f"job {job_id} never reached {status}, is {st}")

    begin = lambda root, preview, sample="NGC-11": req("/api/start", dict(sample=sample, root=root, preview=preview))
    assert req("/api/state", token="wrong")[0] == 403 and req("/api/start", {}, token="")[0] == 403, "token must be required"
    with urllib.request.urlopen(f"{base}/?token={TOKEN}") as x: assert b"Segment splitter" in x.read()
    with urllib.request.urlopen(f"{base}/help?token={TOKEN}") as x: h = x.read().decode()
    assert "How to use the segment splitter" in h and 'src="screenshots/' not in h and "data:image/png;base64," in h, "help must inline its screenshots"
    try: urllib.request.urlopen(f"{base}/help?token=wrong"); raise AssertionError("help must need the token")
    except urllib.error.HTTPError as e: assert e.code == 403
    assert "//" not in PAGE.split("<script>")[1], "no // comments in the page script: one hid the rest of its line (the sample menu) once"
    assert req("/api/state")[1]["samples"] == ["NGC-11", "B"]
    assert begin("abc", True)[0] == 400 and begin("1 2", True)[0] == 400 and begin("1", "yes")[0] == 400, "bad input must be refused"
    assert begin("1", True, "other")[0] == 400, "unknown sample must be refused"
    assert not calls

    assert begin("1", True)[0] == 200; wait(0, "split")  # a passed preview splits without asking
    assert calls.index(("preview", 1)) < calls.index(("execute", 1, False)), "preview first, then the edit with its terminals"
    msg = req("/api/state")[1]["jobs"][0]["message"]
    assert msg.index("11 (50 L2)") < msg.index("10 (5 L2)"), msg
    assert req("/api/act", {"id": 0, "action": "split"})[0] == 409, "a finished job must not split again"
    assert begin("2", False)[0] == 200; wait(1, "split")
    assert ("execute", 2, True) in calls, "no-preview goes straight to the edit"
    assert begin("3", True)[0] == 200; wait(2, "preview_failed")
    assert req("/api/act", {"id": 2, "action": "split"})[0] == 409, "a failed preview offers no split"

    def before(*a): raise bs.BeforeEdit("too small")
    bs.execute = before
    assert begin("4", False)[0] == 200; wait(3, "failed")

    def boom(*a): raise RuntimeError("timeout")
    bs.execute = boom
    assert begin("5", False)[0] == 200; wait(4, "unknown")
    assert begin("5", False)[0] == 409, "unknown outcome must block another edit"
    assert req("/api/act", {"id": 4, "action": "recheck"})[0] == 200; wait(4, "split")
    assert "99" in req("/api/state")[1]["jobs"][4]["message"]

    def idle():
        for _ in range(500):
            if not any(j["status"] in RUNNING for j in req("/api/state")[1]["jobs"]): return time.sleep(.05)  # let a just-finished job add its children
            time.sleep(.02)
        raise AssertionError("jobs never finished")

    def tree(root, preview, rounds):
        n = len(jobs); assert req("/api/start", dict(sample="NGC-11", root=root, preview=preview, rounds=rounds))[0] == 200
        for _ in range(3): idle()
        return jobs[n:]

    for bad in (0, MAX_ROUNDS + 1, "2", True, 1.5): assert req("/api/start", dict(sample="NGC-11", root="6", preview=True, rounds=bad))[0] == 400, bad
    cuts = lambda fn: setattr(bs, "execute", lambda cg, r, p, l, np: (calls.append(("execute", r, np)), dict(op=7, verify_error=None, seconds=0, separated=True) | fn(r))[1])
    two = lambda r: dict(new=[r * 10, r * 10 + 1], sizes=[50, 60])
    cuts(two)

    t = tree("6", True, 2); assert t[0]["status"] == "split" and len(t) == 3, "the first cut splits unasked too"
    assert ("preview", 60) in calls and ("execute", 60, False) in calls and ("execute", 61, False) in calls, "children preview, then split unasked"
    assert not any(c[0] == "execute" and c[1] >= 600 for c in calls), "the last round must not cut again"
    assert [j["parent"] for j in jobs[-2:]] == [t[0]["id"]] * 2 and all(j["rounds"] == 1 for j in jobs[-2:])

    t = tree("7", False, 3); assert len(t) == 7 and all(j["status"] == "split" for j in t), "3 rounds = 7 cuts"
    assert ("execute", 700, True) in calls and ("preview", 70) not in calls, "no-preview trees never preview"

    cuts(lambda r: dict(new=[3, 30], sizes=[50, 60]) if r == 9 else two(r))  # the fake preview fails root 3
    tree("9", True, 2)
    assert ("execute", 3, False) not in calls and ("execute", 30, False) in calls, "a failed preview stops only its branch"

    cuts(lambda r: dict(new=[r * 10, r * 10 + 1, r * 10 + 2], sizes=[5, 50, 2]))
    assert [j["root"] for j in tree("11", False, 2)] == ["11", "111"], "fragments and too-small pieces are not cut"
    cuts(lambda r: dict(two(r), separated=False)); t = tree("12", False, 2)
    assert len(t) == 1 and "not run" in t[0]["message"], "a failed separation stops the branch"
    cuts(lambda r: dict(two(r), verify_error="boom")); assert len(tree("13", False, 2)) == 1, "an unverified cut is not cut again"

    jobs[0]["status"] = "splitting"; jobs[1]["status"] = "previewing"; save(); jobs.clear(); load()
    assert jobs[0]["status"] == "unknown" and jobs[1]["status"] == "failed", "a restart must flag interrupted jobs"
    e400 = RuntimeError("4 attempts, source+sink terminals 46+44; last error: 400 Client Error: BAD REQUEST for url: https://x/split_preview content: "
                        "b'{\\n \"code\": 400,\\n \"message\": \"[400]: Paths irreparably overlap.\",\\n \"timestamp\": \"t\"\\n}\\n'")
    assert brief(e400) == "4 attempts, source+sink terminals 46+44; last error: 400 Client Error: BAD REQUEST ([400]: Paths irreparably overlap.)", brief(e400)
    e500 = RuntimeError("500 Server Error: Invalid vertex index: 9939 for url: https://x content:b'{\\n \"message\": \"Invalid vertex index: 9939\", \"traceback\": [\"...\"]'")
    assert brief(e500) == "500 Server Error: Invalid vertex index: 9939", brief(e500)
    assert brief(RuntimeError("x" * 900)).endswith("(Copy for the full text)")
    bs.preview = lambda cg, r, p: (_ for _ in ()).throw(e500)
    assert begin("20", True)[0] == 200; jid = len(jobs) - 1; wait(jid, "failed")
    j = req("/api/state")[1]["jobs"][jid]; assert j["message"].endswith("9939") and "traceback" in j["detail"], j
    def no_login(sample): raise RuntimeError("401 Client Error: UNAUTHORIZED for url: https://x/info content: b'{\\n  \"error\": \"invalid_token\",\\n  \"message\": \"Unauthorized - Token is Invalid or Expired\"\\n}'")
    connect = no_login
    assert begin("21", False)[0] == 200; jid = len(jobs) - 1; wait(jid, "failed")
    j = req("/api/state")[1]["jobs"][jid]; assert j["message"].startswith("Nothing was edited") and "CAVE token" in j["message"], j
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="anything but localhost sends the token and edits unencrypted")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default=CONFIG_PATH, help="JSON file listing the samples and the CAVE server (default: config.json here)")
    a = ap.parse_args()
    if a.selftest: selftest(); raise SystemExit
    load_config(a.config)
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True); load()
    s = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Open: http://{a.host}:{s.server_address[1]}/?token={TOKEN}", flush=True)
    s.serve_forever()