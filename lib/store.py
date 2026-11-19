# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:50:40 2026

@author: Pedro Gomez Galvez
"""

"""Shared configuration, job persistence, change register, and utility helpers."""
import base64, csv, datetime, io, json, os, re, secrets, threading, concurrent.futures
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
CATMAID_CONFIG_PATH = Path(os.path.join(ROOT_DIR, ".cave_catmaid_config.json"))
STATE_PATH = os.path.join(ROOT_DIR, "outputs", "gui_jobs.json")
CHANGELOG_JSON_PATH = os.path.join(ROOT_DIR, "outputs", "change_register.json")
CHANGELOG_CSV_PATH = os.path.join(ROOT_DIR, "outputs", "change_register.csv")
GUIDE_DIR = os.path.join(ROOT_DIR, "end_user")

SAMPLES, SERVER = {}, ""
RUNNING = {"queued", "previewing", "splitting"}
MAX_ROUNDS = 6
TOKEN = secrets.token_urlsafe(32)

jobs = []
changelog = []
lock = threading.RLock()
pool = concurrent.futures.ThreadPoolExecutor(4)

CHANGELOG_FIELDS = [
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


def load_config(path):
    """Read the samples and server from a JSON config. Fails at startup if unusable."""
    global SERVER
    try:
        with open(path) as f:
            c = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Can't read config {path}: {e}")
    s = c.get("samples") if isinstance(c, dict) else None
    srv = c.get("server_address") if isinstance(c, dict) else None
    if not (isinstance(s, dict) and s and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in s.items())):
        raise SystemExit(f'Config {path}: "samples" must map at least one sample name to a datastack name')
    if not (isinstance(srv, str) and srv.startswith("https://")):
        raise SystemExit(f'Config {path}: "server_address" must be an https:// URL')
    SAMPLES.clear()
    SAMPLES.update(s)
    SERVER = srv


def save():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jobs, f)
    os.replace(tmp, STATE_PATH)


def load():
    if not os.path.exists(STATE_PATH):
        return
    with open(STATE_PATH) as f:
        jobs[:] = json.load(f)
    for j in jobs:
        if j["status"] == "splitting":
            j.update(status="unknown", message="The page stopped during this edit, which may still have applied. Wait a few minutes, then re-check.")
        elif j["status"] in RUNNING:
            j.update(status="failed", message="The page stopped before this finished. Nothing was edited. Start it again.")


def _write_changelog_files():
    os.makedirs(os.path.dirname(CHANGELOG_JSON_PATH), exist_ok=True)
    tmp_json = CHANGELOG_JSON_PATH + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(changelog, f, indent=2)
    os.replace(tmp_json, CHANGELOG_JSON_PATH)

    tmp_csv = CHANGELOG_CSV_PATH + ".tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHANGELOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(changelog)
    os.replace(tmp_csv, CHANGELOG_CSV_PATH)


def load_changelog():
    if not os.path.exists(CHANGELOG_JSON_PATH):
        return
    try:
        with open(CHANGELOG_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            with lock:
                changelog[:] = data
    except Exception:
        pass


def record_change(system, project, target, action, before_ids, after_ids, operation_id="", summary=""):
    """Persistently record a real mutation applied to CAVE or CATMAID."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with lock:
        next_id = max((int(x.get("id", 0)) for x in changelog), default=0) + 1
        entry = {
            "id": next_id,
            "timestamp": now_utc,
            "system": str(system),
            "project": str(project or ""),
            "target": str(target or ""),
            "action": str(action),
            "before_ids": str(before_ids or ""),
            "after_ids": str(after_ids or ""),
            "operation_id": str(operation_id or ""),
            "summary": str(summary or ""),
        }
        changelog.append(entry)
        try:
            _write_changelog_files()
        except Exception:
            pass


def changelog_as_csv():
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CHANGELOG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    with lock:
        writer.writerows(changelog)
    return buf.getvalue()


def brief(e, n=600):
    """Readable error: replaces raw HTTP response body with server's own message."""
    s = str(e)
    m = re.search(r"\s*for url: \S+ content:\s*b'(.*)", s, re.S)
    if m:
        inner = re.search(r'"message":\s*"((?:[^"\\]|\\.)*)"', m.group(1))
        s = s[:m.start()] + (f" ({inner.group(1)})" if inner and inner.group(1) not in s[:m.start()] else "")
    s = s if len(s) <= n else s[:n] + "… (Copy for the full text)"
    if "401" in s and "token" in s.lower():
        s += ' The CAVE token on the server is missing or expired: renew it (README, "CAVE token").'
    return s


def failure(e):
    return dict(detail=str(e)[:20000])


def help_page():
    """end_user/guide.html with its screenshots inlined."""
    with open(os.path.join(GUIDE_DIR, "guide.html"), encoding="utf-8") as f:
        html = f.read()

    def inline(m):
        path = os.path.join(GUIDE_DIR, "screenshots", os.path.basename(m.group(1)))
        if not os.path.exists(path):
            return m.group(0)
        with open(path, "rb") as f:
            return 'src="data:image/png;base64,' + base64.b64encode(f.read()).decode() + '"'

    return re.sub(r'src="screenshots/([\w.-]+\.png)"', inline, html)


def update(job, **kw):
    if "message" in kw:
        kw.setdefault("detail", "")
    with lock:
        job.update(kw)
        save()


def paths(root):
    out = os.path.join(ROOT_DIR, "outputs", root)
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, "preview_terminals.json"), os.path.join(out, "edit_log.csv")