from collections import deque
import inspect
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import navis
import pymaid
from caveclient import CAVEclient

from .spatial_mapper import map_nodes_to_cave


def _fetch_l2_coords_nm(cave_client: CAVEclient, l2_ids: np.ndarray) -> dict[int, np.ndarray]:
    """Query representative 3D coordinates (nm) for L2 IDs when l2cache is available."""
    coords = {}
    id_list = [int(x) for x in l2_ids]
    batch_size = 5000
    for i in range(0, len(id_list), batch_size):
        chunk = id_list[i : i + batch_size]
        info = cave_client.l2cache.get_l2data(chunk, attributes=["rep_coord_nm"])
        for k, v in info.items():
            pt = v.get("rep_coord_nm") if isinstance(v, dict) else None
            if pt is not None and len(pt) == 3:
                coords[int(k)] = np.asarray(pt, dtype=np.float64)
    return coords


def _fetch_cave_mesh(cave_client: CAVEclient, root_id: int):
    """Fetch the 3D surface mesh (vertices in nm) for a CAVE root via CloudVolume."""
    from cloudvolume import CloudVolume

    seg_source = cave_client.info.segmentation_source()
    cv_kwargs = {
        "use_https": True,
        "parallel": 1,
        "progress": False,
    }
    token = getattr(getattr(cave_client, "auth", None), "token", None)
    if token:
        cv_kwargs["secrets"] = token

    vol = CloudVolume(seg_source, **cv_kwargs)
    try:
        raw_mesh = vol.mesh.get(int(root_id), deduplicate_chunk_boundaries=False)
    except TypeError:
        raw_mesh = vol.mesh.get(int(root_id))

    if isinstance(raw_mesh, dict):
        if int(root_id) in raw_mesh:
            return raw_mesh[int(root_id)]
        if raw_mesh:
            return next(iter(raw_mesh.values()))
        raise ValueError(f"CloudVolume returned an empty mesh dictionary for CAVE root {root_id}.")
    return raw_mesh


def _points_to_mst_neuron(
    pts_nm: np.ndarray,
    root_id: int,
    scale: np.ndarray,
    connect_radius_nm: float = 450.0,
    min_twig_nm: float = 250.0,
) -> navis.TreeNeuron:
    """Connect 3D points (nm) into a pruned minimum-spanning-tree SWC neuron."""
    n_nodes = len(pts_nm)
    if n_nodes == 0:
        raise ValueError(f"No 3D coordinates available to skeletonize CAVE root {root_id}.")
    if n_nodes == 1:
        pt = pts_nm[0] / scale
        df = pd.DataFrame([{"node_id": 1, "parent_id": -1, "x": pt[0], "y": pt[1], "z": pt[2]}])
        out = navis.TreeNeuron(df)
        out.id = root_id
        out.name = f"cave_{root_id}"
        return out

    tree = cKDTree(pts_nm)
    pairs = tree.query_pairs(r=connect_radius_nm, output_type="ndarray")
    if len(pairs) == 0:
        k = min(4, n_nodes)
        _, idx = tree.query(pts_nm, k=k)
        edge_list = []
        for i in range(n_nodes):
            for j in idx[i, 1:]:
                edge_list.append((min(i, int(j)), max(i, int(j))))
        pairs = np.unique(np.asarray(edge_list, dtype=np.int32), axis=0)

    u_idx = pairs[:, 0]
    v_idx = pairs[:, 1]
    weights = np.linalg.norm(pts_nm[u_idx] - pts_nm[v_idx], axis=1)

    graph = csr_matrix((weights, (u_idx, v_idx)), shape=(n_nodes, n_nodes))
    mst = minimum_spanning_tree(graph).tocoo()

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for i, j, w in zip(mst.row, mst.col, mst.data):
        adj[int(i)].append((int(j), float(w)))
        adj[int(j)].append((int(i), float(w)))

    removed = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        if len(adj[i]) == 1:
            nbr, dist = adj[i][0]
            if len(adj[nbr]) >= 3 and dist < min_twig_nm:
                removed[i] = True

    parent = np.full(n_nodes, -1, dtype=np.int32)
    visited = removed.copy()
    for start in range(n_nodes):
        if visited[start]:
            continue
        visited[start] = True
        q = deque([start])
        while q:
            curr = q.popleft()
            for nbr, _ in adj[curr]:
                if not visited[nbr]:
                    visited[nbr] = True
                    parent[nbr] = curr
                    q.append(nbr)

    kept_indices = np.where(~removed)[0]
    old_to_new = {int(old_i): new_i + 1 for new_i, old_i in enumerate(kept_indices)}
    pts_cm = pts_nm[kept_indices] / scale

    rows = []
    for new_i, old_i in enumerate(kept_indices):
        p_old = int(parent[old_i])
        p_new = old_to_new.get(p_old, -1)
        rows.append({
            "node_id": new_i + 1,
            "parent_id": p_new,
            "x": float(pts_cm[new_i, 0]),
            "y": float(pts_cm[new_i, 1]),
            "z": float(pts_cm[new_i, 2]),
        })

    out = navis.TreeNeuron(pd.DataFrame(rows))
    out.id = root_id
    out.name = f"cave_{root_id}"
    return out


def skeletonize_cave_root(
    cave_client: CAVEclient,
    root_id: int,
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
    min_twig_nm: float = 250.0,
) -> navis.TreeNeuron:
    """Build an acyclic 3D skeleton from a CAVE segment (via l2cache if running, else CloudVolume mesh)."""
    root_id = int(root_id)
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    if getattr(cave_client, "l2cache", None) is not None:
        try:
            import pcg_skel

            sig_params = inspect.signature(pcg_skel.pcg_skeleton).parameters
            skel_kwargs = {"root_id": root_id, "client": cave_client}
            if "refine" in sig_params:
                skel_kwargs["refine"] = None
            if "invalidation_d" in sig_params:
                skel_kwargs["invalidation_d"] = 500

            sk = pcg_skel.pcg_skeleton(**skel_kwargs)
            verts_cm = np.asarray(sk.vertices, dtype=np.float64) / scale
            if len(verts_cm) > 0:
                parents = np.asarray(sk.parent_nodes(np.arange(len(verts_cm))), dtype=np.int64)
                df = pd.DataFrame({
                    "node_id": np.arange(1, len(verts_cm) + 1, dtype=np.int64),
                    "parent_id": np.where(parents >= 0, parents + 1, -1),
                    "x": verts_cm[:, 0],
                    "y": verts_cm[:, 1],
                    "z": verts_cm[:, 2],
                })
                out = navis.TreeNeuron(df)
                out.id = root_id
                out.name = f"cave_{root_id}"
                return out
        except Exception:
            pass

    mesh = _fetch_cave_mesh(cave_client, root_id)
    verts_nm = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts_nm) == 0:
        raise ValueError(f"CAVE root {root_id} mesh has 0 vertices.")

    try:
        import skeletor as sk

        tm = sk.utilities.make_trimesh(mesh, validate=True)
        to_remove = max(1, int(0.0001 * len(tm.vertices)))
        tm = sk.pre.fix_mesh(tm, inplace=True, remove_disconnected=to_remove)
        s = sk.skeletonize.by_wavefront(tm, waves=1, step_size=1, progress=False)
        swc = s.swc.copy()
        swc["node_id"] = swc["node_id"].astype(np.int64) + 1
        swc["parent_id"] = np.where(swc["parent_id"] >= 0, swc["parent_id"].astype(np.int64) + 1, -1)
        swc[["x", "y", "z"]] = swc[["x", "y", "z"]].to_numpy(dtype=np.float64) / scale
        out = navis.TreeNeuron(swc[["node_id", "parent_id", "x", "y", "z"]])
        out.id = root_id
        out.name = f"cave_{root_id}"
        return out
    except ImportError:
        pass

    voxel_keys = np.floor(verts_nm / 200.0).astype(np.int64)
    _, inv = np.unique(voxel_keys, axis=0, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    centroids_nm = np.column_stack([
        np.bincount(inv, weights=verts_nm[:, 0]) / counts,
        np.bincount(inv, weights=verts_nm[:, 1]) / counts,
        np.bincount(inv, weights=verts_nm[:, 2]) / counts,
    ])
    return _points_to_mst_neuron(centroids_nm, root_id=root_id, scale=scale, min_twig_nm=min_twig_nm)


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
    """Downsample unbranched tracts to ~1 node every `slice_step` Z-slices (interpolated in CATMAID),
    while preserving all roots, branch points, terminal tips, and sharp bends."""
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


def _build_neuron_specific_boxes(
    cave_skeletons: dict[int, navis.TreeNeuron],
    scale: np.ndarray,
    pad_nm: float,
    tile_size_nm: float = 2000.0,
    max_boxes: int = 45,
) -> list[tuple[float, float, float, float, float, float]]:
    """Cover ONLY the CAVE skeletons' actual 3D paths with tight local sub-boxes (padded by pad_nm)
    instead of one giant bounding box, avoiding unrelated neurons across the rest of the neuropil."""
    sample_chunks = []
    for nrn in cave_skeletons.values():
        nodes = nrn.nodes
        pts = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        sample_chunks.append(pts)

        # Densify along skeleton edges so long segments never skip an intermediate grid cell
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

        # Merge contiguous runs along X, then Y, then Z into tight non-overlapping sub-boxes
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

    boxes_nm = []
    for x0, x1, y0, y1, z0, z1 in xyz_runs:
        boxes_nm.append((
            float(x0 * tile_size_nm - pad_nm),
            float((x1 + 1) * tile_size_nm + pad_nm),
            float(y0 * tile_size_nm - pad_nm),
            float((y1 + 1) * tile_size_nm + pad_nm),
            float(z0 * tile_size_nm - pad_nm),
            float((z1 + 1) * tile_size_nm + pad_nm),
        ))
    return boxes_nm


def _fetch_skids_in_local_boxes(
    rm: pymaid.CatmaidInstance,
    boxes_nm: list[tuple[float, float, float, float, float, float]],
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
) -> set[int]:
    """Query CATMAID skeletons overlapping ONLY the neuron-specific local boxes."""
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    skids: set[int] = set()

    for box in boxes_nm:
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
        try:
            url = rm.make_url(rm.project_id, "skeletons", "in-bounding-box", **bbox_params)
            found = rm.fetch(url)
            if isinstance(found, dict):
                found = found.get("skeleton_ids") or found.get("skids") or []
            skids.update(int(s) for s in (found or []))
        except Exception:
            # Fallback to CATMAID's node/list endpoint if skeletons/in-bounding-box is unavailable
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
            treenodes = res[0] if isinstance(res, list) and len(res) > 0 else []
            for tn in treenodes:
                if isinstance(tn, (list, tuple)) and len(tn) > 8 and tn[8] is not None:
                    skids.add(int(tn[8]))

    return skids


def load_manual_catmaid_trees(
    rm: pymaid.CatmaidInstance,
    search_annotations: str = "",
    local_boxes_nm: list[tuple[float, float, float, float, float, float]] | None = None,
    catmaid_resolution_nm: tuple[float, float, float] = (8.0, 8.0, 16.0),
    catmaid_coord_unit: str = "nm",
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Load candidate CATMAID neurons and keep ONLY manually traced nodes (excluding 'cave_imported')."""
    skids: set[int] = set()
    ann_list = [a.strip() for a in search_annotations.split(",") if a.strip()]

    if ann_list:
        for ann in ann_list:
            try:
                found = pymaid.get_skids_by_annotation(ann, remote_instance=rm)
                skids.update(int(s) for s in found)
            except Exception:
                continue
    elif local_boxes_nm:
        skids.update(
            _fetch_skids_in_local_boxes(
                rm,
                boxes_nm=local_boxes_nm,
                catmaid_resolution_nm=catmaid_resolution_nm,
                catmaid_coord_unit=catmaid_coord_unit,
            )
        )

    if not skids:
        return {}, pd.DataFrame()

    neurons = pymaid.get_neurons(list(skids), remote_instance=rm)
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    trees_by_skid: dict[int, dict] = {}
    internal_records = []

    for n in neurons:
        skid = int(n.id)
        nodes = n.nodes.copy()
        tags = n.tags if isinstance(n.tags, dict) else {}
        imported_ids = set(int(nid) for nid in tags.get("cave_imported", []))

        manual = nodes[~nodes["node_id"].isin(imported_ids)].copy()
        if manual.empty:
            continue

        manual["node_id"] = manual["node_id"].astype(int)
        manual["parent_id"] = manual["parent_id"].fillna(-1).astype(int)
        coords_nm = manual[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        manual["x_nm"] = coords_nm[:, 0]
        manual["y_nm"] = coords_nm[:, 1]
        manual["z_nm"] = coords_nm[:, 2]

        trees_by_skid[skid] = {
            "skeleton_id": skid,
            "name": str(n.name or f"neuron_{skid}"),
            "manual_nodes": manual,
        }

        parent_ids = set(manual["parent_id"])
        internal = manual[manual["node_id"].isin(parent_ids)].copy()
        if internal.empty:
            internal = manual.copy()
        internal["skeleton_id"] = skid
        internal_records.append(internal[["skeleton_id", "node_id", "parent_id", "x", "y", "z"]])

    internal_df = pd.concat(internal_records, ignore_index=True) if internal_records else pd.DataFrame()
    return trees_by_skid, internal_df


def _point_to_segments_min_dist(pts_nm: np.ndarray, manual_df: pd.DataFrame) -> np.ndarray:
    """Compute minimum 3D distance (nm) from each point in pts_nm to manual_df's parent->child segments."""
    m_pts = manual_df[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    if len(manual_df) < 2:
        return cdist(pts_nm, m_pts).min(axis=1)

    by_id = manual_df.set_index("node_id")[["x_nm", "y_nm", "z_nm"]]
    valid = manual_df[manual_df["parent_id"].isin(by_id.index)]
    if valid.empty:
        return cdist(pts_nm, m_pts).min(axis=1)

    seg_a = by_id.loc[valid["parent_id"].to_numpy()].to_numpy(dtype=np.float64)
    seg_b = valid[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)

    ab = seg_b - seg_a
    ab_len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    ap = pts_nm[:, None, :] - seg_a[None, :, :]
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
    closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(pts_nm[:, None, :] - closest, axis=2).min(axis=1)


def _split_kept_into_subtrees(
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


def build_cave_export_plan(
    cave_root_ids: list[int],
    cave_client: CAVEclient,
    rm: pymaid.CatmaidInstance,
    cave_resolution_nm: tuple[float, float, float],
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
    search_annotations: str = "",
    cut_buffer_nm: float = 400.0,
    slice_step: int = 4,
    min_fragment_nodes: int = 3,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """Skeletonize CAVE roots, pair with manual CATMAID nodes in the neuron-specific area, cut & decimate."""
    unique_roots = list(dict.fromkeys(int(r) for r in cave_root_ids))
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    cave_skeletons: dict[int, navis.TreeNeuron] = {}
    for idx, rid in enumerate(unique_roots, start=1):
        if progress_callback:
            progress_callback(f"Skeletonizing CAVE root {rid} ({idx}/{len(unique_roots)})...")
        if not cave_client.chunkedgraph.is_latest_roots([rid])[0]:
            latest = cave_client.chunkedgraph.get_latest_roots(rid)
            rid = int(latest[0])
        nrn = skeletonize_cave_root(cave_client, rid, catmaid_resolution_nm, catmaid_coord_unit)
        cave_skeletons[rid] = nrn

    local_boxes_nm = None
    if not search_annotations.strip():
        local_boxes_nm = _build_neuron_specific_boxes(
            cave_skeletons,
            scale=scale,
            pad_nm=float(cut_buffer_nm) + 150.0,
        )
        if progress_callback:
            progress_callback(f"Checking neuron-specific neighborhood ({len(local_boxes_nm)} tight local sub-boxes)...")
    elif progress_callback:
        progress_callback("Loading manual CATMAID skeletons from search annotations...")

    trees_by_skid, internal_df = load_manual_catmaid_trees(
        rm,
        search_annotations=search_annotations,
        local_boxes_nm=local_boxes_nm,
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
    )

    root_to_skids: dict[int, list[tuple[int, int]]] = {}
    if not internal_df.empty:
        mapped_cm = map_nodes_to_cave(
            internal_df,
            cave_client,
            catmaid_resolution_nm=catmaid_resolution_nm,
            cave_resolution_nm=cave_resolution_nm,
            catmaid_coord_unit=catmaid_coord_unit,
        )
        for (r_id, skid), grp in mapped_cm.groupby(["root_id", "skeleton_id"]):
            root_to_skids.setdefault(int(r_id), []).append((int(skid), int(len(grp))))

    plan_rows = []
    bundle_items = {}

    for rid, nrn in cave_skeletons.items():
        total_nodes = int(len(nrn.nodes))
        cave_pts_nm = nrn.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale

        matched_pairs = sorted(root_to_skids.get(rid, []), key=lambda x: -x[1])
        matched_skids = [skid for skid, _ in matched_pairs if skid in trees_by_skid]

        if not matched_skids:
            match_type = "0-Match (New)"
            master_skid = None
            base_name = f"cave_{rid}"
            keep_mask = np.ones(total_nodes, dtype=bool)
            combined_manual = pd.DataFrame()
        else:
            master_skid = matched_skids[0]
            base_name = trees_by_skid[master_skid]["name"]
            match_type = "1-to-1 Match" if len(matched_skids) == 1 else f"1-to-Many ({len(matched_skids)} SKIDs)"
            combined_manual = pd.concat(
                [trees_by_skid[s]["manual_nodes"] for s in matched_skids],
                ignore_index=True,
            )
            min_dists = _point_to_segments_min_dist(cave_pts_nm, combined_manual)
            keep_mask = min_dists > float(cut_buffer_nm)

        raw_subtrees = _split_kept_into_subtrees(nrn, keep_mask, min_fragment_nodes=min_fragment_nodes)
        subtrees = [
            decimate_subtree_by_slices(
                sub,
                catmaid_resolution_nm=catmaid_resolution_nm,
                catmaid_coord_unit=catmaid_coord_unit,
                slice_step=slice_step,
            )
            for sub in raw_subtrees
        ]
        kept_nodes = int(sum(len(s) for s in subtrees))

        n_frags = len(subtrees)
        fragment_specs = []
        for idx_f, sub_df in enumerate(subtrees, start=1):
            if n_frags >= 3:
                frag_name = f"{base_name}_{idx_f}"
            elif n_frags == 2:
                frag_name = f"{base_name}_part{idx_f}"
            else:
                frag_name = base_name

            anchor_info = None
            if not combined_manual.empty:
                sub_pts_nm = sub_df[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
                m_pts_nm = combined_manual[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
                dmat = cdist(sub_pts_nm, m_pts_nm)
                sub_i, man_j = np.unravel_index(np.argmin(dmat), dmat.shape)
                anchor_info = {
                    "sub_local_node_id": int(sub_df.iloc[sub_i]["node_id"]),
                    "master_skid": int(master_skid),
                    "anchor_dist_nm": float(dmat[sub_i, man_j]),
                }

            fragment_specs.append({
                "name": frag_name,
                "nodes_df": sub_df,
                "anchor": anchor_info,
            })

        frag_names_preview = ", ".join(f["name"] for f in fragment_specs) if fragment_specs else "(already complete)"
        plan_rows.append({
            "approve": len(fragment_specs) > 0,
            "cave_root_id": str(rid),
            "match_type": match_type,
            "master_skid": master_skid,
            "paired_skids": ",".join(map(str, matched_skids)) if matched_skids else "None",
            "base_name": base_name,
            "kept_nodes": kept_nodes,
            "total_nodes": total_nodes,
            "fragment_count": len(fragment_specs),
            "fragment_names": frag_names_preview,
        })
        bundle_items[str(rid)] = {
            "master_skid": master_skid,
            "paired_skids": matched_skids,
            "base_name": base_name,
            "fragments": fragment_specs,
        }

    return plan_rows, {
        "items": bundle_items,
        "catmaid_resolution_nm": list(catmaid_resolution_nm),
        "catmaid_coord_unit": catmaid_coord_unit,
    }


def _extract_uploaded_skid(upload_resp) -> int:
    if isinstance(upload_resp, dict) and "skeleton_id" in upload_resp:
        return int(upload_resp["skeleton_id"])
    if isinstance(upload_resp, list) and upload_resp and isinstance(upload_resp[0], dict):
        return int(upload_resp[0]["skeleton_id"])
    raise RuntimeError(f"Unexpected response from pymaid.upload_neuron: {upload_resp}")


def _join_catmaid_skeletons(
    rm: pymaid.CatmaidInstance,
    master_skid: int,
    child_skid: int,
    scale: np.ndarray,
    max_stitch_dist_nm: float = 2000.0,
) -> float:
    """Join child_skid into master_skid on the CATMAID server at their closest 3D nodes."""
    pair = pymaid.get_neurons([int(master_skid), int(child_skid)], remote_instance=rm)
    by_id = {int(n.id): n for n in pair}
    if int(master_skid) not in by_id or int(child_skid) not in by_id:
        raise RuntimeError(f"Could not fetch skeletons {master_skid} and {child_skid} for joining.")

    m_nodes = by_id[int(master_skid)].nodes
    c_nodes = by_id[int(child_skid)].nodes
    m_pts_nm = m_nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
    c_pts_nm = c_nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale

    dmat = cdist(m_pts_nm, c_pts_nm)
    mi, ci = np.unravel_index(np.argmin(dmat), dmat.shape)
    min_dist = float(dmat[mi, ci])
    if min_dist > float(max_stitch_dist_nm):
        raise ValueError(f"gap {min_dist:.0f} nm > {max_stitch_dist_nm:.0f} nm limit")

    master_nid = int(m_nodes.iloc[mi]["node_id"])
    child_nid = int(c_nodes.iloc[ci]["node_id"])

    if hasattr(pymaid, "join_skeletons"):
        pymaid.join_skeletons(
            int(child_skid),
            int(master_skid),
            from_node=child_nid,
            to_node=master_nid,
            remote_instance=rm,
        )
    else:
        url = rm.make_url(rm.project_id, "skeleton", "join")
        rm.fetch(
            url,
            post={
                "from_id": int(child_skid),
                "to_id": int(master_skid),
                "from_node_id": child_nid,
                "to_node_id": master_nid,
            },
        )
    return min_dist


def execute_cave_export_plan(
    plan_rows: list[dict],
    export_bundle: dict,
    rm: pymaid.CatmaidInstance,
    target_annotations: str = "",
    auto_stitch: bool = True,
    max_stitch_dist_nm: float = 2000.0,
    progress_callback=None,
) -> list[dict]:
    """Upload approved non-overlapping CAVE fragments to CATMAID, tag 'cave_imported', stitch & annotate."""
    ann_list = [a.strip() for a in target_annotations.split(",") if a.strip()]
    approved = [r for r in plan_rows if r.get("approve")]
    items = export_bundle.get("items", {})
    catmaid_res = export_bundle.get("catmaid_resolution_nm", [8.0, 8.0, 16.0])
    catmaid_unit = export_bundle.get("catmaid_coord_unit", "nm")
    scale = (
        np.asarray(catmaid_res, dtype=np.float64)
        if catmaid_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )
    audit = []

    for idx, row in enumerate(approved, start=1):
        rid_str = str(row["cave_root_id"])
        item = items.get(rid_str)
        if not item or not item["fragments"]:
            continue

        if progress_callback:
            progress_callback(f"[{idx}/{len(approved)}] Exporting CAVE root {rid_str} to CATMAID...")

        master_skid = item["master_skid"]
        created_skids = []
        created_names = []
        nodes_added = 0
        notes = []

        try:
            for frag in item["fragments"]:
                sub_df = frag["nodes_df"]
                nrn = navis.TreeNeuron(sub_df)
                nrn.name = frag["name"]
                nrn.tags = {"cave_imported": sub_df["node_id"].astype(int).tolist()}

                up_res = pymaid.upload_neuron(
                    nrn,
                    remote_instance=rm,
                    import_tags=True,
                    import_annotations=False,
                    import_connectors=False,
                )
                new_skid = _extract_uploaded_skid(up_res)
                nodes_added += len(sub_df)

                anchor = frag.get("anchor")
                if auto_stitch and master_skid is not None and anchor is not None:
                    try:
                        dist_nm = _join_catmaid_skeletons(
                            rm,
                            master_skid=int(master_skid),
                            child_skid=int(new_skid),
                            scale=scale,
                            max_stitch_dist_nm=max_stitch_dist_nm,
                        )
                        notes.append(f"Stitched {frag['name']} into SKID {master_skid} ({dist_nm:.0f} nm)")
                    except Exception as join_exc:
                        created_skids.append(new_skid)
                        created_names.append(frag["name"])
                        notes.append(f"Kept {frag['name']} (SKID {new_skid}) separate: {join_exc}")
                else:
                    created_skids.append(new_skid)
                    created_names.append(frag["name"])

            if auto_stitch and master_skid is not None and len(item["paired_skids"]) > 1:
                for sec_skid in item["paired_skids"][1:]:
                    try:
                        dist_nm = _join_catmaid_skeletons(
                            rm,
                            master_skid=int(master_skid),
                            child_skid=int(sec_skid),
                            scale=scale,
                            max_stitch_dist_nm=max_stitch_dist_nm,
                        )
                        notes.append(f"Bridged SKID {sec_skid} into Master SKID {master_skid} ({dist_nm:.0f} nm)")
                    except Exception as exc:
                        notes.append(f"Could not bridge SKID {sec_skid}: {exc}")

            final_skids = []
            if master_skid is not None:
                final_skids.append(int(master_skid))
            final_skids.extend(created_skids)
            final_skids = list(dict.fromkeys(final_skids))

            if ann_list and final_skids:
                pymaid.add_annotations(final_skids, ann_list, remote_instance=rm)

            audit.append({
                "cave_root_id": rid_str,
                "status": "SUCCESS",
                "resulting_skids": ",".join(map(str, final_skids)),
                "fragment_names": ", ".join(created_names) if created_names else f"Merged into {item['base_name']}",
                "nodes_added": nodes_added,
                "annotations": ", ".join(ann_list),
                "details": "; ".join(notes) if notes else "Uploaded cleanly",
            })
        except Exception as exc:
            audit.append({
                "cave_root_id": rid_str,
                "status": "FAILED",
                "resulting_skids": ",".join(map(str, created_skids)),
                "fragment_names": ", ".join(created_names),
                "nodes_added": nodes_added,
                "annotations": "",
                "details": str(exc),
            })

    return audit