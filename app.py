"""Web server entry point for Segment Splitter & CATMAID-Driven Proofreading. See README.md."""
import warnings
warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*", category=FutureWarning)

import argparse, hmac, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import splitter as bs
from lib import autosplit_service, catmaid_service, store

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "ui.html")
with open(UI_PATH, encoding="utf-8") as _f:
    PAGE = _f.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # default logging would print the token

    def reply(self, code, body, ctype="application/json"):
        body = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorised(self, given):
        if given and hmac.compare_digest(given.encode(), store.TOKEN.encode()):
            return True
        self.reply(403, {"error": "Missing or wrong token"})
        return False

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            if self.authorised(parse_qs(url.query).get("token", [""])[0]):
                self.reply(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/help":
            if self.authorised(parse_qs(url.query).get("token", [""])[0]):
                try:
                    self.reply(200, store.help_page(), "text/html; charset=utf-8")
                except OSError:
                    self.reply(404, {"error": "The guide (end_user/guide.html) is missing"})
        elif url.path == "/api/state":
            if self.authorised(self.headers.get("X-Token", "")):
                with store.lock:
                    self.reply(
                        200,
                        {
                            "jobs": store.jobs,
                            "samples": list(store.SAMPLES),
                            "max_rounds": store.MAX_ROUNDS,
                            "now": time.time(),
                        },
                    )
        elif url.path == "/api/catmaid/state":
            if self.authorised(self.headers.get("X-Token", "")):
                self.reply(200, catmaid_service.public_state())
        elif url.path == "/api/changelog":
            if self.authorised(self.headers.get("X-Token", "")):
                with store.lock:
                    self.reply(200, {"entries": list(reversed(store.changelog))})
        elif url.path == "/api/changelog.csv":
            if self.authorised(parse_qs(url.query).get("token", [""])[0] or self.headers.get("X-Token", "")):
                csv_txt = store.changelog_as_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="cave_catmaid_change_register.csv"')
                self.send_header("Content-Length", str(len(csv_txt)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(csv_txt)
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorised(self.headers.get("X-Token", "")):
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 2_000_000:
                raise ValueError
            d = json.loads(self.rfile.read(n))
        except ValueError:
            return self.reply(400, {"error": "Bad request"})

        path = urlparse(self.path).path
        routes = {
            "/api/start": lambda x: (lambda c, e: (c, {"error": e}))(*autosplit_service.start(x.get("sample"), x.get("root"), x.get("preview"), x.get("rounds", 1))),
            "/api/act": lambda x: (lambda c, e: (c, {"error": e}))(*autosplit_service.act(x.get("id"), x.get("action"))),
            "/api/catmaid/probe": catmaid_service.probe_resolutions,
            "/api/catmaid/save_project": catmaid_service.save_project,
            "/api/catmaid/plan": catmaid_service.start_plan,
            "/api/catmaid/preview_split": catmaid_service.preview_single_split,
            "/api/catmaid/execute": catmaid_service.start_execute,
            "/api/catmaid/export_plan": catmaid_service.start_export_plan,
            "/api/catmaid/export_execute": catmaid_service.start_export_exec,
        }
        fn = routes.get(path)
        if not fn:
            return self.reply(404, {"error": "Not found"})
        code, payload = fn(d)
        self.reply(code, payload)


def selftest():
    """Whole flow over real HTTP against a fake chunkedgraph. Makes no CAVE calls."""
    import tempfile, urllib.error, urllib.request
    bs.selftest()
    d = tempfile.mkdtemp()
    store.STATE_PATH = os.path.join(d, "jobs.json")
    store.CHANGELOG_JSON_PATH = os.path.join(d, "change_register.json")
    store.CHANGELOG_CSV_PATH = os.path.join(d, "change_register.csv")
    store.changelog.clear()
    cfg = os.path.join(d, "config.json")
    for bad in ("{", '{"samples": {}, "server_address": "https://x"}', '{"samples": {"A": "a"}, "server_address": "http://x"}'):
        with open(cfg, "w") as f:
            f.write(bad)
        try:
            store.load_config(cfg)
            raise AssertionError(f"bad config must be refused: {bad}")
        except SystemExit:
            pass
    with open(cfg, "w") as f:
        json.dump({"server_address": "https://x", "samples": {"NGC-11": "ngc11_datastack", "B": "b"}}, f)
    store.load_config(cfg)
    assert store.SERVER == "https://x" and store.SAMPLES == {"NGC-11": "ngc11_datastack", "B": "b"}

    class FakeCG:
        def is_latest_roots(self, r):
            return [False]

        def get_latest_roots(self, r):
            return [99]

    calls = []
    autosplit_service.connect = lambda sample: (calls.append(("connect", sample)), FakeCG())[1]
    store.paths = lambda root: ("p", "l")
    bs.preview = lambda cg, r, p: (
        calls.append(("preview", r)),
        dict(passed=r != 3, legal=True, source_l2=1, sink_l2=1, both_l2=0, neither_l2=0, missing_sv=0, total_sv=9, links=0, seconds=0),
    )[1]
    bs.execute = lambda cg, r, p, l, np: (
        calls.append(("execute", r, np)),
        dict(op=7, new=[r * 10, r * 10 + 1], sizes=[5, 50], separated=True, verify_error=None, seconds=0),
    )[1]
    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{s.server_address[1]}"

    def req(path, body=None, token=store.TOKEN):
        r = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers={"X-Token": token})
        try:
            with urllib.request.urlopen(r) as x:
                return x.status, json.loads(x.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(job_id, status):
        for _ in range(200):
            st = req("/api/state")[1]["jobs"][job_id]["status"]
            if st == status:
                return
            time.sleep(.02)
        raise AssertionError(f"job {job_id} never reached {status}, is {st}")

    begin = lambda root, preview, sample="NGC-11": req("/api/start", dict(sample=sample, root=root, preview=preview))
    assert req("/api/state", token="wrong")[0] == 403 and req("/api/start", {}, token="")[0] == 403
    with urllib.request.urlopen(f"{base}/?token={store.TOKEN}") as x:
        assert b"Segment splitter" in x.read()
    with urllib.request.urlopen(f"{base}/help?token={store.TOKEN}") as x:
        h = x.read().decode()
    assert "How to use" in h and 'src="screenshots/' not in h
    try:
        urllib.request.urlopen(f"{base}/help?token=wrong")
        raise AssertionError("help must need the token")
    except urllib.error.HTTPError as e:
        assert e.code == 403
    assert "//" not in PAGE.split("<script>")[1]
    assert req("/api/state")[1]["samples"] == ["NGC-11", "B"]
    assert begin("abc", True)[0] == 400 and begin("1 2", True)[0] == 400 and begin("1", "yes")[0] == 400
    assert begin("1", True, "other")[0] == 400
    assert not calls

    assert begin("1", True)[0] == 200
    wait(0, "split")
    assert calls.index(("preview", 1)) < calls.index(("execute", 1, False))
    msg = req("/api/state")[1]["jobs"][0]["message"]
    assert msg.index("11 (50 L2)") < msg.index("10 (5 L2)"), msg
    assert req("/api/act", {"id": 0, "action": "split"})[0] == 409
    assert begin("2", False)[0] == 200
    wait(1, "split")
    assert ("execute", 2, True) in calls
    assert begin("3", True)[0] == 200
    wait(2, "preview_failed")
    assert req("/api/act", {"id": 2, "action": "split"})[0] == 409

    def before(*a):
        raise bs.BeforeEdit("too small")

    bs.execute = before
    assert begin("4", False)[0] == 200
    wait(3, "failed")

    def boom(*a):
        raise RuntimeError("timeout")

    bs.execute = boom
    assert begin("5", False)[0] == 200
    wait(4, "unknown")
    assert begin("5", False)[0] == 409
    assert req("/api/act", {"id": 4, "action": "recheck"})[0] == 200
    wait(4, "split")
    assert "99" in req("/api/state")[1]["jobs"][4]["message"]

    def idle():
        for _ in range(500):
            if not any(j["status"] in store.RUNNING for j in req("/api/state")[1]["jobs"]):
                return time.sleep(.05)
            time.sleep(.02)
        raise AssertionError("jobs never finished")

    def tree(root, preview, rounds):
        n = len(store.jobs)
        assert req("/api/start", dict(sample="NGC-11", root=root, preview=preview, rounds=rounds))[0] == 200
        for _ in range(3):
            idle()
        return store.jobs[n:]

    for bad in (0, store.MAX_ROUNDS + 1, "2", True, 1.5):
        assert req("/api/start", dict(sample="NGC-11", root="6", preview=True, rounds=bad))[0] == 400
    cuts = lambda fn: setattr(bs, "execute", lambda cg, r, p, l, np: (calls.append(("execute", r, np)), dict(op=7, verify_error=None, seconds=0, separated=True) | fn(r))[1])
    two = lambda r: dict(new=[r * 10, r * 10 + 1], sizes=[50, 60])
    cuts(two)

    t = tree("6", True, 2)
    assert t[0]["status"] == "split" and len(t) == 3
    assert ("preview", 60) in calls and ("execute", 60, False) in calls and ("execute", 61, False) in calls
    assert not any(c[0] == "execute" and c[1] >= 600 for c in calls)
    assert [j["parent"] for j in store.jobs[-2:]] == [t[0]["id"]] * 2 and all(j["rounds"] == 1 for j in store.jobs[-2:])

    t = tree("7", False, 3)
    assert len(t) == 7 and all(j["status"] == "split" for j in t)
    assert ("execute", 700, True) in calls and ("preview", 70) not in calls

    cuts(lambda r: dict(new=[3, 30], sizes=[50, 60]) if r == 9 else two(r))
    tree("9", True, 2)
    assert ("execute", 3, False) not in calls and ("execute", 30, False) in calls

    cuts(lambda r: dict(new=[r * 10, r * 10 + 1, r * 10 + 2], sizes=[5, 50, 2]))
    assert [j["root"] for j in tree("11", False, 2)] == ["11", "111"]
    cuts(lambda r: dict(two(r), separated=False))
    t = tree("12", False, 2)
    assert len(t) == 1 and "not run" in t[0]["message"]
    cuts(lambda r: dict(two(r), verify_error="boom"))
    assert len(tree("13", False, 2)) == 1

    store.jobs[0]["status"] = "splitting"
    store.jobs[1]["status"] = "previewing"
    store.save()
    store.jobs.clear()
    store.load()
    assert store.jobs[0]["status"] == "unknown" and store.jobs[1]["status"] == "failed"
    e400 = RuntimeError(
        "4 attempts, source+sink terminals 46+44; last error: 400 Client Error: BAD REQUEST for url: https://x/split_preview content: "
        "b'{\\n \"code\": 400,\\n \"message\": \"[400]: Paths irreparably overlap.\",\\n \"timestamp\": \"t\"\\n}\\n'"
    )
    assert store.brief(e400) == "4 attempts, source+sink terminals 46+44; last error: 400 Client Error: BAD REQUEST ([400]: Paths irreparably overlap.)"
    e500 = RuntimeError("500 Server Error: Invalid vertex index: 9939 for url: https://x content:b'{\\n \"message\": \"Invalid vertex index: 9939\", \"traceback\": [\"...\"]'")
    assert store.brief(e500) == "500 Server Error: Invalid vertex index: 9939"
    assert store.brief(RuntimeError("x" * 900)).endswith("(Copy for the full text)")
    bs.preview = lambda cg, r, p: (_ for _ in ()).throw(e500)
    assert begin("20", True)[0] == 200
    jid = len(store.jobs) - 1
    wait(jid, "failed")
    j = req("/api/state")[1]["jobs"][jid]
    assert j["message"].endswith("9939") and "traceback" in j["detail"]

    def no_login(sample):
        raise RuntimeError(
            "401 Client Error: UNAUTHORIZED for url: https://x/info content: "
            "b'{\\n  \"error\": \"invalid_token\",\\n  \"message\": \"Unauthorized - Token is Invalid or Expired\"\\n}'"
        )

    autosplit_service.connect = no_login
    assert begin("21", False)[0] == 200
    jid = len(store.jobs) - 1
    wait(jid, "failed")
    j = req("/api/state")[1]["jobs"][jid]
    assert j["message"].startswith("Nothing was edited") and "CAVE token" in j["message"]
    assert len(req("/api/changelog")[1]["entries"]) > 0
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="anything but localhost sends the token and edits unencrypted")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default=store.CONFIG_PATH, help="JSON file listing the samples and the CAVE server")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    store.load_config(a.config)
    os.makedirs(os.path.dirname(store.STATE_PATH), exist_ok=True)
    store.load()
    store.load_changelog()
    s = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Open: http://{a.host}:{s.server_address[1]}/?token={store.TOKEN}", flush=True)
    s.serve_forever()