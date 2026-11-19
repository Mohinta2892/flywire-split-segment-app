"""CATMAID local-neighborhood queries, cKDTree segment distance, subtree splitting, and Z-slice decimation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect

import navis
import numpy as np
import pandas as pd
import pymaid
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .retry import retry_call

_BBOX_ENDPOINT_SUPPORTED: dict[str, bool] = {}


@contextmanager
def _no_pymaid_cache(rm: pymaid.CatmaidInstance):
    """Temporarily disable pymaid's HTTP cache to avoid high per-query overhead in batch fetches."""
    has_attr = hasattr(rm, "caching")
    prev = getattr(rm, "caching", None)
    if has_attr:
        rm.caching = False
    try:
        yield
    finally:
        if has_attr:
            rm.caching = prev


def _get_neurons_fast(rm: pymaid.CatmaidInstance, skids: list[int]):
    """Fetch skeleton geometries from CATMAID without connectors/tags/history/cache overhead."""
    sig = inspect.signature(pymaid.get_neurons)
    kw = {"remote_instance": rm}
    for opt, val in (
        ("with_connectors", False),
        ("with_tags", False),
        ("with_history", False),
        ("hide_progress", True),
    ):
        if opt in sig.parameters:
            kw[opt] = val
    with _no_pymaid_cache(rm):
        return pymaid.get_neurons(skids, **kw)


def _max_chord_deviation_nm(pts_nm: np.ndarray, idx_a: int, idx_b: int) -> float:
    """Maximum perpendicular 3D distance (nm) of intermediate points between idx_a and idx_b."""
    if idx_b - idx_a <= 1:
        return 0.0
    a = pts_nm[idx_a]
    b = pts_nm[idx_b]
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    mid = pts_nm[idx_a + 1 : idx_b]
    if ab_len2 < 1e-12:
        return float(np.linalg.norm(mid - a, axis=1).max())
    t = np.clip(((mid - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.linalg.norm(mid - proj, axis=1).max())


def decimate_subtree_by_slices(
    sub_df: pd.DataFrame,
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
    slice_step: int = 4,
    max_chord_dev_nm: float = 150.0,
) -> pd.DataFrame:
    """Downsample unbranched tracts to ~1 node every `slice_step` Z-slices while preserving topology and bends."""
    slice_step = int(slice_step)
    if slice_step <= 1 or len(sub_df) <= 3:
        return sub_df

    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    z_res_nm = max(float(catmaid_resolution_nm[2]), 1.0)
    step_dist_nm = slice_step * z_res_nm

    node_ids = sub_df["node_id"].astype(int).tolist()
    parent_ids = sub_df["parent_id"].astype(int).tolist()
    coords_cm = sub_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    coords_nm = coords_cm * scale

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    children: dict[int, list[int]] = {i: [] for i in range(len(node_ids))}
    parent_idx = np.full(len(node_ids), -1, dtype=np.int32)

    for i, pid in enumerate(parent_ids):
        if pid in id_to_idx:
            p_i = id_to_idx[pid]
            parent_idx[i] = p_i
            children[p_i].append(i)

    is_critical = np.array(
        [(parent_idx[i] == -1) or (len(children[i]) != 1) for i in range(len(node_ids))],
        dtype=bool,
    )

    new_parent = parent_idx.copy()
    keep_node = is_critical.copy()

    for u in np.where(is_critical)[0]:
        for c in children[u]:
            chain = [u, c]
            curr = c
            while not is_critical[curr]:
                curr = children[curr][0]
                chain.append(curr)

            if len(chain) <= 2:
                continue

            pts = coords_nm[chain]
            kept_pos = [0]
            cum_path = 0.0

            for k in range(1, len(chain) - 1):
                cum_path += float(np.linalg.norm(pts[k] - pts[k - 1]))
                last_p = kept_pos[-1]

                if k - last_p > 1 and _max_chord_deviation_nm(pts, last_p, k) > max_chord_dev_nm:
                    kept_pos.append(k - 1)
                    last_p = k - 1
                    cum_path = float(np.linalg.norm(pts[k] - pts[k - 1]))

                dz_slices = abs(pts[k, 2] - pts[last_p, 2]) / z_res_nm
                if dz_slices >= slice_step or cum_path >= step_dist_nm:
                    kept_pos.append(k)
                    cum_path = 0.0

            end_pos = len(chain) - 1
            if len(kept_pos) > 1:
                dist_to_end = float(np.linalg.norm(pts[end_pos] - pts[kept_pos[-1]]))
                dz_from_prev = abs(pts[end_pos, 2] - pts[kept_pos[-2], 2]) / z_res_nm
                if dist_to_end < 0.4 * step_dist_nm and dz_from_prev <= (slice_step + 1):
                    kept_pos.pop()

            kept_pos.append(end_pos)

            for pos_idx in range(1, len(kept_pos)):
                child_node = chain[kept_pos[pos_idx]]
                par_node = chain[kept_pos[pos_idx - 1]]
                keep_node[child_node] = True
                new_parent[child_node] = par_node

    kept_indices = np.where(keep_node)[0]
    old_to_new = {int(old_i): new_i + 1 for new_i, old_i in enumerate(kept_indices)}

    rows = []
    for new_i, old_i in enumerate(kept_indices):
        p_old = int(new_parent[old_i])
        rows.append({
            "node_id": new_i + 1,
            "parent_id": old_to_new.get(p_old, -1),
            "x": float(coords_cm[old_i, 0]),
            "y": float(coords_cm[old_i, 1]),
            "z": float(coords_cm[old_i, 2]),
        })

    return pd.DataFrame(rows)


def build_neuron_specific_boxes(
    cave_skeletons: dict[int, navis.TreeNeuron],
    scale: np.ndarray,
    pad_nm: float,
    tile_size_nm: float = 4000.0,
    max_boxes: int = 16,
) -> list[tuple[float, float, float, float, float, float]]:
    """Cover ONLY the CAVE skeletons' actual 3D paths with tight local sub-boxes (padded by pad_nm)."""
    sample_chunks = []
    for nrn in cave_skeletons.values():
        nodes = nrn.nodes
        pts = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        sample_chunks.append(pts)

        by_id = nodes.set_index("node_id")[["x", "y", "z"]]
        valid = nodes[nodes["parent_id"].isin(by_id.index)]
        if not valid.empty:
            a = by_id.loc[valid["parent_id"].to_numpy()].to_numpy(dtype=np.float64) * scale
            b = valid[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
            dists = np.linalg.norm(b - a, axis=1)
            for idx_e in np.where(dists > 500.0)[0]:
                n_steps = int(np.ceil(dists[idx_e] / 500.0))
                ts = np.linspace(0.0, 1.0, n_steps + 1)[1:-1, None]
                sample_chunks.append(a[idx_e] + ts * (b[idx_e] - a[idx_e]))

    all_pts = np.vstack(sample_chunks)

    while True:
        cells = np.unique(np.floor(all_pts / tile_size_nm).astype(np.int64), axis=0)
        cells = cells[np.lexsort((cells[:, 0], cells[:, 1], cells[:, 2]))]

        x_runs = []
        for ix, iy, iz in cells:
            if x_runs and x_runs[-1][2] == iy and x_runs[-1][3] == iz and x_runs[-1][1] + 1 == ix:
                x_runs[-1][1] = int(ix)
            else:
                x_runs.append([int(ix), int(ix), int(iy), int(iz)])

        x_runs.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
        xy_runs = []
        for x0, x1, iy, iz in x_runs:
            if xy_runs and xy_runs[-1][0] == x0 and xy_runs[-1][1] == x1 and xy_runs[-1][4] == iz and xy_runs[-1][3] + 1 == iy:
                xy_runs[-1][3] = iy
            else:
                xy_runs.append([x0, x1, iy, iy, iz])

        xy_runs.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
        xyz_runs = []
        for x0, x1, y0, y1, iz in xy_runs:
            if (
                xyz_runs
                and xyz_runs[-1][0] == x0
                and xyz_runs[-1][1] == x1
                and xyz_runs[-1][2] == y0
                and xyz_runs[-1][3] == y1
                and xyz_runs[-1][5] + 1 == iz
            ):
                xyz_runs[-1][5] = iz
            else:
                xyz_runs.append([x0, x1, y0, y1, iz, iz])

        if len(xyz_runs) <= max_boxes:
            break
        tile_size_nm *= 1.5

    return [
        (
            float(x0 * tile_size_nm - pad_nm),
            float((x1 + 1) * tile_size_nm + pad_nm),
            float(y0 * tile_size_nm - pad_nm),
            float((y1 + 1) * tile_size_nm + pad_nm),
            float(z0 * tile_size_nm - pad_nm),
            float((z1 + 1) * tile_size_nm + pad_nm),
        )
        for x0, x1, y0, y1, z0, z1 in xyz_runs
    ]


def _fetch_skids_in_local_boxes(
    rm: pymaid.CatmaidInstance,
    boxes_nm: list[tuple[float, float, float, float, float, float]],
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
) -> set[int]:
    """Query CATMAID skeletons overlapping ONLY the neuron-specific local boxes in parallel."""
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    srv_key = str(getattr(rm, "server", "default"))

    def _query_node_list(bbox_params):
        url = rm.make_url(rm.project_id, "node", "list")
        res = rm.fetch(
            url,
            post={
                "left": bbox_params["minx"],
                "top": bbox_params["miny"],
                "z1": bbox_params["minz"],
                "right": bbox_params["maxx"],
                "bottom": bbox_params["maxy"],
                "z2": bbox_params["maxz"],
            },
        )
        out_ids = set()
        treenodes = res[0] if isinstance(res, list) and len(res) > 0 else []
        for tn in treenodes:
            if isinstance(tn, (list, tuple)) and len(tn) > 8 and tn[8] is not None:
                out_ids.add(int(tn[8]))
        return out_ids

    def _query_box(box):
        b_min = np.array([box[0], box[2], box[4]]) / scale
        b_max = np.array([box[1], box[3], box[5]]) / scale
        bbox_params = {
            "minx": float(b_min[0]),
            "miny": float(b_min[1]),
            "minz": float(b_min[2]),
            "maxx": float(b_max[0]),
            "maxy": float(b_max[1]),
            "maxz": float(b_max[2]),
        }
        if _BBOX_ENDPOINT_SUPPORTED.get(srv_key) is False:
            return _query_node_list(bbox_params)

        try:
            url = rm.make_url(rm.project_id, "skeletons", "in-bounding-box", **bbox_params)
            found = rm.fetch(url)
            if isinstance(found, dict):
                found = found.get("skeleton_ids") or found.get("skids") or []
            _BBOX_ENDPOINT_SUPPORTED[srv_key] = True
            return {int(s) for s in (found or [])}
        except Exception:
            _BBOX_ENDPOINT_SUPPORTED[srv_key] = False
            return _query_node_list(bbox_params)

    skids: set[int] = set()
    with _no_pymaid_cache(rm):
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(boxes_nm)))) as ex:
            for batch_ids in ex.map(_query_box, boxes_nm):
                skids.update(batch_ids)
    return skids


def load_existing_catmaid_trees(
    rm: pymaid.CatmaidInstance,
    search_annotations: str = "",
    local_boxes_nm: list[tuple[float, float, float, float, float, float]] | None = None,
    catmaid_resolution_nm: tuple[float, float, float] = (8.0, 8.0, 16.0),
    catmaid_coord_unit: str = "nm",
    progress_callback=None,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Load candidate CATMAID neurons and their existing traced nodes."""
    skids: set[int] = set()
    ann_list = [a.strip() for a in search_annotations.split(",") if a.strip()]

    if ann_list:
        with _no_pymaid_cache(rm):
            for ann in ann_list:
                try:
                    found = retry_call(
                        lambda a=ann: pymaid.get_skids_by_annotation(a, remote_instance=rm),
                        stage=f"CATMAID annotation '{ann}'",
                        progress_callback=progress_callback,
                    )
                    skids.update(int(s) for s in found)
                except Exception:
                    continue
    elif local_boxes_nm:
        skids.update(
            retry_call(
                lambda: _fetch_skids_in_local_boxes(
                    rm,
                    boxes_nm=local_boxes_nm,
                    catmaid_resolution_nm=catmaid_resolution_nm,
                    catmaid_coord_unit=catmaid_coord_unit,
                ),
                stage="CATMAID local bounding-box query",
                progress_callback=progress_callback,
            )
        )

    if not skids:
        return {}, pd.DataFrame()

    if progress_callback:
        try:
            progress_callback(f"Downloading {len(skids)} candidate CATMAID skeleton(s)...", 72.0)
        except TypeError:
            progress_callback(f"Downloading {len(skids)} candidate CATMAID skeleton(s)...")

    neurons = retry_call(
        lambda: _get_neurons_fast(rm, list(skids)),
        stage="CATMAID get_neurons",
        progress_callback=progress_callback,
    )
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    trees_by_skid: dict[int, dict] = {}
    internal_records = []

    for n in neurons:
        skid = int(n.id)
        existing = n.nodes.copy()
        if existing.empty:
            continue

        existing["node_id"] = existing["node_id"].astype(int)
        existing["parent_id"] = existing["parent_id"].fillna(-1).astype(int)
        coords_nm = existing[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        existing["x_nm"] = coords_nm[:, 0]
        existing["y_nm"] = coords_nm[:, 1]
        existing["z_nm"] = coords_nm[:, 2]

        trees_by_skid[skid] = {
            "skeleton_id": skid,
            "name": str(n.name or f"neuron_{skid}"),
            "existing_nodes": existing,
        }

        parent_ids = set(existing["parent_id"])
        internal = existing[existing["node_id"].isin(parent_ids)].copy()
        if internal.empty:
            internal = existing.copy()

        internal["skeleton_id"] = skid
        internal_records.append(internal[["skeleton_id", "node_id", "parent_id", "x", "y", "z", "x_nm", "y_nm", "z_nm"]])

    internal_df = pd.concat(internal_records, ignore_index=True) if internal_records else pd.DataFrame()
    return trees_by_skid, internal_df


def point_to_segments_min_dist(
    pts_nm: np.ndarray,
    existing_df: pd.DataFrame,
    max_query_radius_nm: float | None = None,
) -> np.ndarray:
    """Compute minimum 3D distance (nm) from pts_nm to existing_df segments using cKDTree spatial prefiltering."""
    m_pts = existing_df[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    if len(existing_df) < 2:
        return cdist(pts_nm, m_pts).min(axis=1)

    by_id = existing_df.set_index("node_id")[["x_nm", "y_nm", "z_nm"]]
    valid = existing_df[existing_df["parent_id"].isin(by_id.index)]
    if valid.empty:
        return cdist(pts_nm, m_pts).min(axis=1)

    seg_a = by_id.loc[valid["parent_id"].to_numpy()].to_numpy(dtype=np.float64)
    seg_b = valid[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    ab = seg_b - seg_a
    ab_len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)

    if max_query_radius_nm is None or len(seg_a) <= 256:
        ap = pts_nm[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
        return np.linalg.norm(pts_nm[:, None, :] - closest, axis=2).min(axis=1)

    midpoints = 0.5 * (seg_a + seg_b)
    half_lens = 0.5 * np.sqrt(ab_len2)
    search_r = float(max_query_radius_nm) + float(half_lens.max())
    tree = cKDTree(midpoints)
    cand_lists = tree.query_ball_point(pts_nm, r=search_r)

    out = np.full(len(pts_nm), np.inf, dtype=np.float64)
    for i, cands in enumerate(cand_lists):
        if not cands:
            continue
        c_arr = np.asarray(cands, dtype=np.int32)
        sa = seg_a[c_arr]
        ab_c = ab[c_arr]
        len2_c = ab_len2[c_arr]
        ap = pts_nm[i] - sa
        t = np.clip(np.sum(ap * ab_c, axis=1) / len2_c, 0.0, 1.0)
        closest = sa + t[:, None] * ab_c
        out[i] = float(np.linalg.norm(pts_nm[i] - closest, axis=1).min())
    return out


def split_kept_into_subtrees(
    cave_nrn: navis.TreeNeuron,
    keep_mask: np.ndarray,
    min_fragment_nodes: int = 3,
) -> list[pd.DataFrame]:
    """Split a CAVE skeleton's non-overlapping nodes into connected subtree DataFrames."""
    nodes = cave_nrn.nodes.copy().reset_index(drop=True)
    kept_idx = np.where(keep_mask)[0]
    if len(kept_idx) == 0:
        return []

    id_to_pos = {int(nid): idx for idx, nid in enumerate(nodes["node_id"])}
    kept_set = set(int(nid) for nid in nodes.loc[kept_idx, "node_id"])

    rows_ep, cols_ep = [], []
    for idx in kept_idx:
        nid = int(nodes.loc[idx, "node_id"])
        pid = int(nodes.loc[idx, "parent_id"])
        if pid in kept_set:
            p_idx = id_to_pos[pid]
            rows_ep.extend([idx, p_idx])
            cols_ep.extend([p_idx, idx])

    n_all = len(nodes)
    adj = csr_matrix((np.ones(len(rows_ep)), (rows_ep, cols_ep)), shape=(n_all, n_all))
    _, labels = connected_components(adj, directed=False)

    subtrees = []
    for comp_id in np.unique(labels[kept_idx]):
        comp_indices = [i for i in kept_idx if labels[i] == comp_id]
        if len(comp_indices) < min_fragment_nodes:
            continue

        sub = nodes.iloc[comp_indices][["node_id", "parent_id", "x", "y", "z"]].copy()
        sub_ids = set(int(x) for x in sub["node_id"])
        remap = {int(old_id): new_i + 1 for new_i, old_id in enumerate(sub["node_id"])}
        sub["node_id"] = [remap[int(x)] for x in sub["node_id"]]
        sub["parent_id"] = [
            remap[int(p)] if int(p) in sub_ids else -1 for p in sub["parent_id"]
        ]
        subtrees.append(sub.reset_index(drop=True))

    subtrees.sort(key=len, reverse=True)
    return subtrees