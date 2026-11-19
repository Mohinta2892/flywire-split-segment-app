# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:51:23 2026

@author: Pedro Gomez Galvez
"""

"""Tab 1: Length-based recursive binary segment splitter workers."""
import time
import splitter as bs
from lib import store


def connect(sample):
    from caveclient import CAVEclient
    return CAVEclient(datastack_name=store.SAMPLES[sample], server_address=store.SERVER).chunkedgraph


def add(sample, root, rounds, parent, preview):
    """Record a job and queue it. A previewed job splits without asking if its preview passes."""
    with store.lock:
        j = dict(
            id=len(store.jobs),
            sample=sample,
            root=root,
            status="queued",
            message="",
            started=time.time(),
            rounds=rounds,
            parent=parent,
        )
        store.jobs.append(j)
        store.save()
    if preview:
        store.pool.submit(run_preview, j)
    else:
        store.pool.submit(run_split, j, True)


def not_run(j):
    left = j.get("rounds", 1) - 1
    return f" {left} more round(s) not run on this branch." if left else ""


def run_preview(j):
    store.update(j, status="previewing", started=time.time())
    try:
        p = bs.preview(connect(j["sample"]), int(j["root"]), store.paths(j["root"])[0])
        msg = (
            f"Would give {p['source_l2']} / {p['sink_l2']} L2 ({p['both_l2']} on both sides, {p['neither_l2']} on neither). "
            f"Missing supervoxels {p['missing_sv']} of {p['total_sv']}, links between sides {p['links']}, {p['seconds']} s."
        )
        store.update(j, status="preview_passed" if p["passed"] else "preview_failed", message=msg)
    except Exception as e:
        store.update(j, status="failed", message=f"Preview failed, nothing was edited: {store.brief(e)}", **store.failure(e))
    else:
        if p["passed"]:
            run_split(j, False)


def run_split(j, no_preview):
    store.update(j, status="splitting", started=time.time(), message="")
    try:
        cg = connect(j["sample"])
    except Exception as e:
        return store.update(j, status="failed", message=f"Nothing was edited, couldn't connect to CAVE: {store.brief(e)}", **store.failure(e))
    try:
        r = bs.execute(cg, int(j["root"]), *store.paths(j["root"]), no_preview)
    except bs.BeforeEdit as e:
        return store.update(j, status="failed", message=f"Nothing was edited: {store.brief(e)}", **store.failure(e))
    except Exception as e:
        return store.update(
            j,
            status="unknown",
            message=f"The edit may still have applied: {store.brief(e)}. Wait a few minutes, then re-check.",
            **store.failure(e),
        )
    sized = len(r["sizes"]) == len(r["new"])
    pieces = sorted(zip(r["new"], r["sizes"]), key=lambda x: -x[1]) if sized else [(x, None) for x in r["new"]]
    msg = f"Op {r['op']}, {r['seconds']} s: " + ", ".join(f"{x} ({n} L2)" if n is not None else str(x) for x, n in pieces) + "."
    if r["verify_error"]:
        msg += f" The edit applied, but its checks failed: {r['verify_error']}"
    clean = sized and r["separated"] is True and not r["verify_error"]
    if not clean:
        msg += not_run(j)
    store.update(j, status="not_separated" if r["separated"] is False else "split", message=msg)
    store.record_change(
        system="CAVE",
        project=j["sample"],
        target=store.SAMPLES.get(j["sample"], ""),
        action="Auto Split",
        before_ids=j["root"],
        after_ids=", ".join(str(x) for x, _ in pieces),
        operation_id=str(r.get("op", "")),
        summary=msg,
    )
    if clean and j.get("rounds", 1) > 1:
        for root, n in pieces[:2]:
            if n >= bs.MIN_L2:
                add(j["sample"], str(root), j["rounds"] - 1, j["id"], not no_preview)


def run_recheck(j):
    try:
        cg = connect(j["sample"])
        if cg.is_latest_roots([int(j["root"])])[0]:
            store.update(j, status="failed", message="The root hasn't changed. If the server is still working, the edit could still apply.")
        else:
            latest_str = ", ".join(str(int(x)) for x in cg.get_latest_roots(int(j["root"])))
            msg = "The edit applied. Latest roots: " + latest_str + "." + not_run(j)
            store.update(j, status="split", message=msg)
            store.record_change(
                system="CAVE",
                project=j["sample"],
                target=store.SAMPLES.get(j["sample"], ""),
                action="Auto Split (Confirmed)",
                before_ids=j["root"],
                after_ids=latest_str,
                operation_id="",
                summary=msg,
            )
    except Exception as e:
        store.update(j, status="unknown", message=f"Re-check failed: {store.brief(e)}", **store.failure(e))


def start(sample, root, preview, rounds=1):
    """-> (http code, error message)."""
    if sample not in store.SAMPLES:
        return 400, "Unknown sample"
    if not (isinstance(root, str) and root.isascii() and root.isdigit() and 0 < int(root) < 2**64):
        return 400, "Enter one segment ID, as a number"
    if not isinstance(preview, bool):
        return 400, "Bad request"
    if type(rounds) is not int or not 1 <= rounds <= store.MAX_ROUNDS:
        return 400, f"Rounds must be a whole number from 1 to {store.MAX_ROUNDS}"
    root = str(int(root))
    with store.lock:
        for j in store.jobs:
            if j["root"] == root and j["status"] in store.RUNNING:
                return 409, "This segment is already running"
            if j["root"] == root and j["status"] == "unknown":
                return 409, "This segment's last edit has an unknown outcome. Re-check it first."
        add(sample, root, rounds, None, preview)
    return 200, ""


def act(job_id, action):
    with store.lock:
        j = next((x for x in store.jobs if x["id"] == job_id), None)
        if not j:
            return 404, "Unknown job"
        if action == "split" and j["status"] == "preview_passed":
            store.update(j, status="queued", started=time.time())
            store.pool.submit(run_split, j, False)
        elif action == "recheck" and j["status"] == "unknown":
            store.pool.submit(run_recheck, j)
        else:
            return 409, f"Can't {action} a job in this state"
    return 200, ""