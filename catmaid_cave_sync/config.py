import json
from pathlib import Path
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


def query_cave_resolution(cave_client: CAVEclient) -> list[float]:
    """Query CAVE segmentation voxel resolution (nm) directly from the server."""
    res = cave_client.info.segmentation_resolution()
    return [float(v) for v in res]


def query_catmaid_stacks(rm: pymaid.CatmaidInstance) -> list[dict]:
    """Query available image stacks and their voxel resolutions (nm) from CATMAID."""
    stacks_df = pymaid.get_stacks(remote_instance=rm)
    stacks = []
    for _, row in stacks_df.iterrows():
        raw_res = row.get("resolution")
        if isinstance(raw_res, dict):
            res = [float(raw_res["x"]), float(raw_res["y"]), float(raw_res["z"])]
        elif isinstance(raw_res, (list, tuple)) and len(raw_res) == 3:
            res = [float(v) for v in raw_res]
        else:
            continue
        stacks.append({
            "stack_id": int(row["id"]),
            "title": str(row.get("title", f"Stack {row['id']}")),
            "resolution_nm": res,
        })
    return stacks


def init_clients_from_project(project_cfg: dict) -> tuple[CAVEclient, pymaid.CatmaidInstance]:
    """Initialize CAVE (using saved local CAVE secret) & CATMAID clients from a project profile."""
    cave_kwargs = {"datastack_name": project_cfg["cave_datastack"]}
    if project_cfg.get("cave_server_address"):
        cave_kwargs["server_address"] = project_cfg["cave_server_address"]

    cave_client = CAVEclient(**cave_kwargs)
    rm = pymaid.CatmaidInstance(
        server=project_cfg["catmaid_url"],
        api_token=project_cfg["catmaid_token"],
        project_id=int(project_cfg["catmaid_project_id"]),
    )
    return cave_client, rm