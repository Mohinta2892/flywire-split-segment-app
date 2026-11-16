import json
from pathlib import Path
import requests
from caveclient import CAVEclient
import pymaid

DEFAULT_CONFIG_PATH = Path(".cave_catmaid_config.json")


def load_all_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the multi-project configuration store from disk."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"active_project": None, "projects": {}}


def save_project_config(
    project_name: str,
    project_cfg: dict,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """Save or update a named project profile (e.g., 'Parker-Goog') locally."""
    store = load_all_configs(config_path)
    store["active_project"] = project_name
    store["projects"][project_name] = project_cfg
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    return project_cfg


def normalize_catmaid_url(url: str) -> str:
    """Strip any '?pid=...' query string or trailing slash from a pasted CATMAID URL."""
    return url.split("?")[0].strip().rstrip("/")


def query_cave_resolution(cave_client: CAVEclient) -> list[float]:
    """Query CAVE segmentation base voxel resolution (nm) from chunkedgraph or info service."""
    if getattr(cave_client, "chunkedgraph", None) is not None:
        base_res = getattr(cave_client.chunkedgraph, "base_resolution", None)
        if base_res is not None and len(base_res) == 3:
            return [float(v) for v in base_res]

    viewer_res = cave_client.info.viewer_resolution()
    return [float(v) for v in viewer_res]


def probe_catmaid_connection(
    server_url: str,
    project_id: int,
    api_token: str,
    http_user: str = "",
    http_password: str = "",
) -> list[dict]:
    """Verify CATMAID authentication & project access, and fetch stack metadata/resolutions."""
    base_url = normalize_catmaid_url(server_url)
    pid = int(project_id)

    headers = {"X-Authorization": f"Token {api_token.strip()}"}
    auth = (http_user.strip(), http_password) if http_user.strip() else None
    resp = requests.get(f"{base_url}/{pid}/stacks", headers=headers, auth=auth, timeout=20)

    if resp.status_code in (401, 403):
        www_auth = resp.headers.get("WWW-Authenticate", "")
        if "basic" in www_auth.lower():
            raise PermissionError(
                "Phase 1 failed (401 HTTP Basic Auth): front-door login rejected the "
                "HTTP User / HTTP Password."
            )
        raise PermissionError(
            f"Phase 1 passed, but Phase 2 failed ({resp.status_code} CATMAID Auth): "
            "check your personal CATMAID API Token and Project ID."
        )
    resp.raise_for_status()

    raw_stacks = resp.json() if isinstance(resp.json(), list) else []
    stacks = []
    for item in raw_stacks:
        stack_id = int(item["id"])
        title = str(item.get("title", f"Stack {stack_id}"))
        res = None
        try:
            s_resp = requests.get(
                f"{base_url}/{pid}/stack/{stack_id}/info",
                headers=headers,
                auth=auth,
                timeout=20,
            )
            if s_resp.ok:
                info = s_resp.json()
                raw_res = info.get("resolution") if isinstance(info, dict) else None
                if isinstance(raw_res, dict):
                    res = [float(raw_res["x"]), float(raw_res["y"]), float(raw_res["z"])]
                elif isinstance(raw_res, (list, tuple)) and len(raw_res) == 3:
                    res = [float(v) for v in raw_res]
        except Exception:
            res = None

        stacks.append({
            "stack_id": stack_id,
            "title": title,
            "resolution_nm": res,
        })

    return stacks


def init_clients_from_project(project_cfg: dict) -> tuple[CAVEclient, pymaid.CatmaidInstance]:
    """Initialize CAVE (using saved local CAVE secret) & CATMAID clients from a project profile."""
    cave_kwargs = {"datastack_name": project_cfg["cave_datastack"]}
    if project_cfg.get("cave_server_address"):
        cave_kwargs["server_address"] = project_cfg["cave_server_address"]

    cave_client = CAVEclient(**cave_kwargs)

    cm_kwargs = {
        "server": normalize_catmaid_url(project_cfg["catmaid_url"]),
        "api_token": project_cfg["catmaid_token"],
        "project_id": int(project_cfg["catmaid_project_id"]),
    }
    if project_cfg.get("catmaid_http_user"):
        cm_kwargs["http_user"] = project_cfg["catmaid_http_user"]
    if project_cfg.get("catmaid_http_password"):
        cm_kwargs["http_password"] = project_cfg["catmaid_http_password"]

    rm = pymaid.CatmaidInstance(**cm_kwargs)
    return cave_client, rm