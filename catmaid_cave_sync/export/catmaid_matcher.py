"""CATMAID skeleton loading, bounding-box neighborhood queries, overlap subtraction, and subtree formatting."""
from collections import deque
from contextlib import contextmanager
import inspect

import navis
import numpy as np
import pandas as pd
import pymaid
from scipy.spatial import cKDTree

from .retry import retry_call


@contextmanager
def _no_pymaid_cache(rm: pymaid.CatmaidInstance):
    """Temporarily disable pymaid HTTP caching so fresh server state is always read/written."""
    prev = getattr(rm, "caching", None)
    try:
        if prev is not None:
            rm.caching = False
        yield
    finally:
        if prev is not None:
            rm.caching = prev


def _get_neurons_fast(
    rm: pymaid.CatmaidInstance,
    skids: list[int],
    batch_size: int = 40,
    progress_callback=None,
) -> list[navis.TreeNeuron]:
    """Fetch CATMAID skeletons in safe chunks without tags/connectors/history to avoid server timeouts."""
    unique_skids = list(dict.fromkeys(int(s) for s in skids))
    if not unique_skids:
        return []

    sig = inspect.signature(pymaid.get_neurons)
    kw: dict = {"remote_instance": rm}
    if "tag_flag" in sig.parameters:
        kw["tag_flag"] = False
    if "connector_flag" in sig.parameters:
        kw["connector_flag"] = False
    if "get_history" in sig.parameters:
        kw["get_history"] = False
    if "raise_missing" in sig.parameters:
        kw["raise_missing"] = False

    neurons: list[navis.TreeNeuron] = []
    with _no_pymaid_cache(rm):
        for start in range(0, len(unique_skids), batch_size):
            chunk = unique_skids[start : start + batch_size]
            res = retry_call(
                lambda c=chunk: pymaid.get_neurons(c, **kw),
                stage=f"CATMAID get_neurons ({start + 1}..{start + len(chunk)}/{len(unique_skids)})",
                progress_callback=progress_callback,
            )
            if isinstance(res, navis.NeuronList):
                neurons.extend(list(res))
            elif isinstance(res, navis.TreeNeuron):
                neurons.append(res)
            elif isinstance(res, (list, tuple)):
                neurons.extend(res)
    return neurons


def build_neuron_specific_boxes(
    cave_skeletons: dict[int, navis.TreeNeuron],
    scale: np.ndarray,
    pad_nm: float = 550.0,
    max_boxes_per_neuron: int = 12,
) -> list[tuple[float, float, float, float, float, float]]:
    """Build tight 3D nanometer bounding boxes around skeleton segments to avoid querying whole-brain slabs."""
    boxes: list[tuple[float, float, float, float, float, float]] = []
    pad = float(pad_nm)
    for nrn in cave_skeletons.values():
        pts = nrn.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        if pts.size == 0:
            continue
        order = np.argsort(pts[:, 2])
        pts_sorted = pts[order]
        n_chunks = min(max_boxes_per_neuron, max(1, int(np.ceil(len(pts_sorted) / 80.0))))
        for chunk in np.array_split(pts_sorted, n_chunks):
            if chunk.size == 0:
                continue
            lo = chunk.min(axis=0) - pad
            hi = chunk.max(axis=0) + pad
            boxes.append((float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])))
    return boxes


def load_existing_catmaid_trees(
    rm: pymaid.CatmaidInstance,
    search_annotations: str = "",
    local_boxes_nm: list[tuple[float, float, float, float, float, float]] | None = None,
    catmaid_resolution_nm: tuple[float, float, float] = (8.0, 8.0, 16.0),
    catmaid_coord_unit: str = "nm",
    progress_callback=None,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Load candidate existing CATMAID skeletons either by annotation or via local 3D bounding boxes."""
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    skids_set: set[int] = set()

    with _no_pymaid_cache(rm):
        anns = [a.strip() for a in (search_annotations or "").split(",") if a.strip()]
        if anns:
            found = retry_call(
                lambda: pymaid.get_skids_by_annotation(anns, remote_instance=rm),
                stage="CATMAID get_skids_by_annotation",
                progress_callback=progress_callback,
            )
            skids_set.update(int(s) for s in found)
        elif local_boxes_nm:
            url = rm.make_url(rm.project_id, "skeletons", "in-bounding-box")
            for box in local_boxes_nm:
                minx, miny, minz, maxx, maxy, maxz = box
                try:
                    res = retry_call(
                        lambda b=(minx, miny, minz, maxx, maxy, maxz): rm.fetch(
                            url,
                            post={
                                "minx": b[0],
                                "miny": b[1],
                                "minz": b[2],
                                "maxx": b[3],
                                "maxy": b[4],
                                "maxz": b[5],
                            },
                        ),
                        retries=2,
                        stage="CATMAID in-bounding-box",
                        progress_callback=progress_callback,
                    )
                    if isinstance(res, list):
                        skids_set.update(int(s) for s in res)
                except Exception:
                    continue

        if not skids_set:
            return {}, pd.DataFrame(columns=["skeleton_id", "node_id", "x", "y", "z", "x_nm", "y_nm", "z_nm"])

        skids = sorted(skids_set)
        names_map: dict[int, str] = {}
        try:
            names_map = {int(k): str(v) for k, v in pymaid.get_names(skids, remote_instance=rm).items()}
        except Exception:
            pass

        neurons = _get_neurons_fast(rm, skids, batch_size=40, progress_callback=progress_callback)

    trees_by_skid: dict[int, dict] = {}
    internal_rows: list[pd.DataFrame] = []

    for nrn in neurons:
        skid = int(nrn.id)
        nodes = nrn.nodes
        if nodes.empty:
            continue
        pts_nm = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        existing_df = pd.DataFrame({
            "node_id": nodes["node_id"].astype(np.int64).to_numpy(),
            "parent_id": nodes["parent_id"].fillna(-1).astype(np.int64).to_numpy(),
            "x_nm": pts_nm[:, 0],
            "y_nm": pts_nm[:, 1],
            "z_nm": pts_nm[:, 2],
        })
        trees_by_skid[skid] = {
            "name": names_map.get(skid) or getattr(nrn, "name", None) or f"skid_{skid}",
            "existing_nodes": existing_df,
        }

        parent_set = set(existing_df["parent_id"].tolist())
        internal = nodes[nodes["node_id"].isin(parent_set) & (nodes["parent_id"] >= 0)]
        if len(internal) < 3:
            internal = nodes
        int_pts_nm = internal[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        internal_rows.append(
            pd.DataFrame({
                "skeleton_id": skid,
                "node_id": internal["node_id"].astype(np.int64).to_numpy(),
                "x": internal["x"].to_numpy(dtype=np.float64),
                "y": internal["y"].to_numpy(dtype=np.float64),
                "z": internal["z"].to_numpy(dtype=np.float64),
                "x_nm": int_pts_nm[:, 0],
                "y_nm": int_pts_nm[:, 1],
                "z_nm": int_pts_nm[:, 2],
            })
        )

    internal_df = (
        pd.concat(internal_rows, ignore_index=True)
        if internal_rows
        else pd.DataFrame(columns=["skeleton_id", "node_id", "x", "y", "z", "x_nm", "y_nm", "z_nm"])
    )
    return trees_by_skid, internal_df


def point_to_segments_min_dist(
    query_pts_nm: np.ndarray,
    existing_nodes_df: pd.DataFrame,
    max_query_radius_nm: float = 500.0,
) -> np.ndarray:
    """Compute exact 3D Euclidean distance from each query point to the nearest CATMAID skeleton edge."""
    n_q = len(query_pts_nm)
    if n_q == 0 or existing_nodes_df.empty:
        return np.full(n_q, np.inf, dtype=np.float64)

    pts = existing_nodes_df[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    nids = existing_nodes_df["node_id"].to_numpy(dtype=np.int64)
    pids = existing_nodes_df["parent_id"].to_numpy(dtype=np.int64)
    skids = (
        existing_nodes_df["skeleton_id"].to_numpy(dtype=np.int64)
        if "skeleton_id" in existing_nodes_df.columns
        else np.zeros(len(pts), dtype=np.int64)
    )

    id_map = {(int(s), int(nid)): i for i, (s, nid) in enumerate(zip(skids, nids))}
    seg_a_list = []
    seg_b_list = []
    sample_chunks = [pts]

    for i, (s, pid) in enumerate(zip(skids, pids)):
        p_idx = id_map.get((int(s), int(pid)))
        if p_idx is not None:
            a = pts[p_idx]
            b = pts[i]
            seg_a_list.append(a)
            seg_b_list.append(b)
            seg_len = float(np.linalg.norm(b - a))
            if seg_len > 120.0:
                steps = int(np.ceil(seg_len / 120.0))
                ts = np.linspace(0.0, 1.0, steps + 1)[1:-1, None]
                sample_chunks.append(a + ts * (b - a))

    tree = cKDTree(np.vstack(sample_chunks))
    approx_dists, _ = tree.query(query_pts_nm, k=1, distance_upper_bound=float(max_query_radius_nm) + 250.0)
    if not seg_a_list:
        return approx_dists

    seg_a = np.asarray(seg_a_list, dtype=np.float64)
    seg_b = np.asarray(seg_b_list, dtype=np.float64)
    seg_vec = seg_b - seg_a
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)
    seg_mid = 0.5 * (seg_a + seg_b)
    seg_half_len = 0.5 * np.sqrt(seg_len2)
    mid_tree = cKDTree(seg_mid)
    max_half = float(seg_half_len.max()) if len(seg_half_len) else 0.0

    out = approx_dists.copy()
    cand_indices = np.where(np.isfinite(approx_dists))[0]
    for qi in cand_indices:
        q = query_pts_nm[qi]
        s_idxs = mid_tree.query_ball_point(q, r=float(max_query_radius_nm) + max_half + 50.0)
        if not s_idxs:
            continue
        a = seg_a[s_idxs]
        ab = seg_vec[s_idxs]
        l2 = seg_len2[s_idxs]
        t = np.where(l2 > 1e-12, np.sum((q - a) * ab, axis=1) / np.maximum(l2, 1e-12), 0.0)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(q - proj, axis=1)
        out[qi] = float(np.min(d))
    return out


def split_kept_into_subtrees(
    nrn: navis.TreeNeuron,
    keep_mask: np.ndarray,
    min_fragment_nodes: int = 3,
) -> list[pd.DataFrame]:
    """Partition kept nodes into connected subtrees, each re-rooted with contiguous 1..K node_ids."""
    nodes = nrn.nodes.reset_index(drop=True)
    n = len(nodes)
    if n == 0 or len(keep_mask) != n:
        return []

    kept_indices = np.where(np.asarray(keep_mask, dtype=bool))[0]
    if len(kept_indices) < max(1, int(min_fragment_nodes)):
        return []

    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    parent_ids = nodes["parent_id"].fillna(-1).to_numpy(dtype=np.int64)
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    kept_set = set(int(i) for i in kept_indices)

    # Undirected adjacency among kept nodes so every connected component becomes 1 single-root tree
    adj: dict[int, list[int]] = {i: [] for i in kept_set}
    orig_parent: dict[int, int] = {}
    for i in kept_set:
        p = id_to_idx.get(int(parent_ids[i]), -1)
        if p in kept_set and p != i:
            adj[i].append(p)
            adj[p].append(i)
            orig_parent[i] = p

    visited: set[int] = set()
    subtrees: list[pd.DataFrame] = []

    for start in kept_indices:
        start = int(start)
        if start in visited:
            continue

        # Collect connected component
        comp: list[int] = []
        q = deque([start])
        visited.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)

        if len(comp) < max(1, int(min_fragment_nodes)):
            continue

        # Pick original root of this component (node whose parent wasn't kept)
        roots = [u for u in comp if u not in orig_parent]
        root_u = roots[0] if roots else comp[0]

        # BFS spanning tree from root_u so parent_id is strictly a single-root DAG
        order: list[int] = [root_u]
        tree_parent: dict[int, int] = {root_u: -1}
        seen = {root_u}
        bfs = deque([root_u])
        while bfs:
            u = bfs.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    tree_parent[v] = u
                    order.append(v)
                    bfs.append(v)

        new_id_map = {u: new_id for new_id, u in enumerate(order, start=1)}
        sub_coords = nodes.iloc[order][["x", "y", "z"]].to_numpy(dtype=np.float64)
        new_nids = np.arange(1, len(order) + 1, dtype=np.int64)
        new_pids = np.array(
            [-1 if tree_parent[u] < 0 else new_id_map[tree_parent[u]] for u in order],
            dtype=np.int64,
        )
        sub_df = pd.DataFrame({
            "node_id": new_nids,
            "parent_id": new_pids,
            "x": sub_coords[:, 0],
            "y": sub_coords[:, 1],
            "z": sub_coords[:, 2],
        })
        subtrees.append(sub_df)

    subtrees.sort(key=len, reverse=True)
    return subtrees


def decimate_subtree_by_slices(
    sub_df: pd.DataFrame,
    catmaid_resolution_nm: tuple[float, float, float] = (8.0, 8.0, 16.0),
    catmaid_coord_unit: str = "nm",
    slice_step: int = 1,
) -> pd.DataFrame:
    """Optionally thin nodes every N Z-slices while preserving branch points, leaves, and horizontal XY runs."""
    step = max(1, int(slice_step))
    if step <= 1 or len(sub_df) <= 3:
        return sub_df.copy(deep=True).reset_index(drop=True)

    z_res = float(catmaid_resolution_nm[2])
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    max_seg_nm = step * z_res * 1.5

    n = len(sub_df)
    nids = sub_df["node_id"].to_numpy(dtype=np.int64)
    pids = sub_df["parent_id"].to_numpy(dtype=np.int64)
    coords = sub_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    coords_nm = coords * scale

    id_to_idx = {int(nid): i for i, nid in enumerate(nids)}
    parent_idx = np.array([id_to_idx.get(int(pid), -1) for pid in pids], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent_idx):
        if p >= 0:
            children[p].append(i)

    keep = np.zeros(n, dtype=bool)
    new_parent_idx = np.full(n, -1, dtype=np.int64)

    # Top-down traversal preserving root, branch points, leaves, Z-step intervals, and max 3D segment length
    roots = [i for i in range(n) if parent_idx[i] < 0]
    for r in roots:
        keep[r] = True
        stack: list[tuple[int, int, float]] = [(c, r, 0.0) for c in children[r]]
        while stack:
            u, last_kept, acc_dist = stack.pop()
            step_dist = float(np.linalg.norm(coords_nm[u] - coords_nm[parent_idx[u]]))
            acc_dist += step_dist
            z_slice_u = int(round(coords_nm[u, 2] / max(z_res, 1e-6)))
            z_slice_k = int(round(coords_nm[last_kept, 2] / max(z_res, 1e-6)))

            is_critical = (parent_idx[u] < 0) or (len(children[u]) != 1)
            hit_z_step = abs(z_slice_u - z_slice_k) >= step
            hit_xy_span = acc_dist >= max_seg_nm

            if is_critical or hit_z_step or hit_xy_span:
                keep[u] = True
                new_parent_idx[u] = last_kept
                next_kept = u
                next_dist = 0.0
            else:
                next_kept = last_kept
                next_dist = acc_dist

            for c in children[u]:
                stack.append((c, next_kept, next_dist))

    kept_indices = np.where(keep)[0]
    remap = {int(old_i): new_i for new_i, old_i in enumerate(kept_indices, start=1)}
    out_pids = np.array(
        [-1 if new_parent_idx[old_i] < 0 else remap[int(new_parent_idx[old_i])] for old_i in kept_indices],
        dtype=np.int64,
    )
    out_coords = coords[kept_indices]
    return pd.DataFrame({
        "node_id": np.arange(1, len(kept_indices) + 1, dtype=np.int64),
        "parent_id": out_pids,
        "x": out_coords[:, 0],
        "y": out_coords[:, 1],
        "z": out_coords[:, 2],
    })