from collections.abc import Sequence
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from cloudvolume import CloudVolume


def _validate_resolution(res: Sequence[float], label: str) -> np.ndarray:
    if res is None or len(res) != 3 or any(float(v) <= 0 for v in res):
        raise ValueError(f"Invalid {label} resolution: {res}. Expected 3 positive values in nm.")
    return np.array(res, dtype=np.float64)


def map_nodes_to_cave(
    nodes_df: pd.DataFrame,
    cave_client: CAVEclient,
    catmaid_resolution_nm: Sequence[float],
    cave_resolution_nm: Sequence[float],
    catmaid_coord_unit: str = "nm",
    root_batch_size: int = 10_000,
) -> pd.DataFrame:
    """Convert CATMAID node coordinates to CAVE voxels using project-specific resolutions."""
    if nodes_df.empty:
        return nodes_df.copy()

    catmaid_res = _validate_resolution(catmaid_resolution_nm, "CATMAID")
    cave_res = _validate_resolution(cave_resolution_nm, "CAVE")

    df = nodes_df.copy()
    raw_coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    # Convert to nanometers if CATMAID coordinates were stored in voxel units
    coords_nm = raw_coords * catmaid_res if catmaid_coord_unit == "voxels" else raw_coords

    # Convert nanometers to CAVE base voxel indices
    coords_vx = np.round(coords_nm / cave_res).astype(np.int64)
    df["x_nm"], df["y_nm"], df["z_nm"] = coords_nm[:, 0], coords_nm[:, 1], coords_nm[:, 2]
    df["vx_x"], df["vx_y"], df["vx_z"] = coords_vx[:, 0], coords_vx[:, 1], coords_vx[:, 2]

    seg_source = cave_client.info.segmentation_source()
    cv_kwargs = {
        "use_https": True,
        "agglomerate": False,
        "lru_bytes": 250_000_000,
        "progress": False,
    }
    # Forward the active CAVE token so custom servers (e.g., braininbrain.org) authenticate cleanly
    cave_token = getattr(cave_client.auth, "token", None)
    if cave_token:
        cv_kwargs["secrets"] = {"token": cave_token}

    cv = CloudVolume(seg_source, **cv_kwargs)

    unique_vx = list({tuple(pt) for pt in coords_vx.tolist()})
    sv_lookup = cv.scattered_points(unique_vx)
    df["supervoxel_id"] = [int(sv_lookup.get(tuple(pt), 0)) for pt in coords_vx.tolist()]

    # Remove background (0) supervoxels
    df = df[df["supervoxel_id"] != 0].reset_index(drop=True)
    if df.empty:
        return df

    # Batch root lookups so 200+ skeletons resolve quickly
    svs = df["supervoxel_id"].to_numpy(dtype=np.int64)
    roots = np.empty(len(svs), dtype=np.int64)
    for start in range(0, len(svs), root_batch_size):
        end = start + root_batch_size
        roots[start:end] = cave_client.chunkedgraph.get_roots(svs[start:end])

    df["root_id"] = roots
    return df