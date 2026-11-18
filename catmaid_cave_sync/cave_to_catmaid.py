import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import navis
import pymaid
from caveclient import CAVEclient

from .spatial_mapper import map_nodes_to_cave


DEFAULT_KIMIMARO_CONFIG = {
    "mip": 3,
    "scale": 1.0,
    "const": 10.0,
    "pdrf_scale": 10000,
    "pdrf_exponent": 2,
    "pdrf_soma_acceptance_threshold": 30,
    "pdrf_soma_detection_threshold": 20,
    "pdrf_soma_invalidation_const": 30,
    "pdrf_soma_invalidation_scale": 1,
    "max_paths": None,
    "dust_threshold": 0,
    "parallel": 4,
    "smooth_window": 2,
    "resample_to": 128.0,
}

_KIMIMARO_WORKER_CODE = """
import json, sys
import numpy as np
import kimimaro

seg_path, cfg_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
seg = np.load(seg_path)
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

skels = kimimaro.skeletonize(
    seg,
    teasar_params={
        "scale": cfg["scale"],
        "const": cfg["const"],
        "pdrf_scale": cfg["pdrf_scale"],
        "pdrf_exponent": cfg["pdrf_exponent"],
        "soma_acceptance_threshold": cfg["pdrf_soma_acceptance_threshold"],
        "soma_detection_threshold": cfg["pdrf_soma_detection_threshold"],
        "soma_invalidation_const": cfg["pdrf_soma_invalidation_const"],
        "soma_invalidation_scale": cfg["pdrf_soma_invalidation_scale"],
        "max_paths": cfg["max_paths"],
    },
    dust_threshold=cfg["dust_threshold"],
    fix_branching=True,
    fix_borders=True,
    fill_holes=False,
    fix_avocados=False,
    progress=False,
    parallel=int(cfg["parallel"]),
    parallel_chunk_size=1,
)

swc_map = {str(int(rid)): skel.to_swc() for rid, skel in skels.items()}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(swc_map, f)
"""


def _extract_host(exc: Exception) -> str:
    txt = str(exc)
    idx = txt.find("http")
    if idx != -1:
        end = idx
        while end < len(txt) and not txt[end].isspace() and txt[end] not in ("'", '"', ")"):
            end += 1
        host = urlparse(txt[idx:end]).netloc
        if host:
            return host
    return ""


def _retry_call(fn, stage: str, max_attempts: int = 4, base_delay: float = 1.5, progress_callback=None):
    """Retry transient HTTP 429/502/503/504 or connection errors with exponential backoff."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(
                code in msg
                for code in ("503", "502", "504", "429", "temporarily unavailable", "bad gateway", "timed out", "connection reset")
            )
            if not is_transient or attempt == max_attempts:
                host = _extract_host(exc)
                where = f"{stage} ({host})" if host else stage
                raise RuntimeError(f"[{where}] {exc}") from exc
            wait_s = base_delay * (2 ** (attempt - 1))
            if progress_callback:
                progress_callback(
                    f"Server temporarily busy during {stage} (attempt {attempt}/{max_attempts}); retrying in {wait_s:.1f}s..."
                )
            time.sleep(wait_s)
    raise last_exc


def _compute_tight_mip_bbox(
    cave_client: CAVEclient,
    root_ids: list[int],
    vol_seg,
    pad_chunks: int = 1,
    progress_callback=None,
):
    """Return a minimal chunk-aligned CloudVolume Bbox covering root_ids at vol_seg.mip."""
    from cloudvolume.lib import Bbox

    full_bounds = vol_seg.bounds
    full_min = np.asarray(full_bounds.minpt, dtype=np.int64)
    full_max = np.asarray(full_bounds.maxpt, dtype=np.int64)

    pts_nm = []
    for rid in root_ids:
        got_root_pts = False
        try:
            l2_ids = _retry_call(
                lambda r=int(rid): cave_client.chunkedgraph.get_leaves(r, stop_layer=2),
                stage=f"CAVE get_leaves({rid})",
                progress_callback=progress_callback,
            )
            if len(l2_ids) > 0 and getattr(cave_client, "l2cache", None) is not None:
                id_list = [int(x) for x in l2_ids]
                for i in range(0, len(id_list), 5000):
                    batch = id_list[i : i + 5000]
                    info = _retry_call(
                        lambda b=batch: cave_client.l2cache.get_l2data(b, attributes=["rep_coord_nm"]),
                        stage="CAVE l2cache",
                        progress_callback=progress_callback,
                    )
                    for v in info.values():
                        pt = v.get("rep_coord_nm") if isinstance(v, dict) else None
                        if pt is not None and len(pt) == 3:
                            pts_nm.append(pt)
                            got_root_pts = True
        except Exception:
            got_root_pts = False

        if got_root_pts:
            continue

        try:
            mesh = _retry_call(
                lambda r=int(rid): vol_seg.mesh.get(r, deduplicate_chunk_boundaries=False),
                stage=f"CAVE mesh({rid})",
                progress_callback=progress_callback,
            )
            m_obj = mesh[int(rid)] if isinstance(mesh, dict) else mesh
            verts = np.asarray(m_obj.vertices, dtype=np.float64)
            if len(verts) > 0:
                pts_nm.append(verts.min(axis=0))
                pts_nm.append(verts.max(axis=0))
                got_root_pts = True
        except Exception:
            pass

        if not got_root_pts:
            return full_bounds

    if not pts_nm:
        return full_bounds

    pts_arr = np.asarray(pts_nm, dtype=np.float64)
    mip_res = np.asarray(vol_seg.resolution, dtype=np.float64)
    chunk_sz = np.maximum(np.asarray(getattr(vol_seg, "chunk_size", [64, 64, 64])[:3], dtype=np.int64), 1)

    raw_min = np.floor(pts_arr.min(axis=0) / mip_res).astype(np.int64)
    raw_max = np.ceil(pts_arr.max(axis=0) / mip_res).astype(np.int64)

    rel_min = np.floor((raw_min - full_min) / chunk_sz).astype(np.int64) - pad_chunks
    rel_max = np.ceil((raw_max - full_min) / chunk_sz).astype(np.int64) + pad_chunks

    min_vox = np.maximum(full_min + rel_min * chunk_sz, full_min)
    max_vox = np.minimum(full_min + rel_max * chunk_sz, full_max)
    max_vox = np.maximum(max_vox, min_vox + 1)
    return Bbox(min_vox.tolist(), max_vox.tolist())


def _run_kimimaro_parallel_swc(seg_subset: np.ndarray, cfg: dict) -> dict[int, str]:
    """Run kimimaro.skeletonize with full multi-CPU parallelization even when called from a worker thread."""
    parallel_workers = max(1, int(cfg.get("parallel", 4)))

    # When called from a background thread, delegate to a dedicated child Python process whose main thread
    # owns signal handlers, allowing kimimaro's multi-process worker pool to run at full speed.
    if parallel_workers > 1 and threading.current_thread() is not threading.main_thread():
        with tempfile.TemporaryDirectory(prefix="kimimaro_par_") as tmpdir:
            seg_path = os.path.join(tmpdir, "seg.npy")
            cfg_path = os.path.join(tmpdir, "cfg.json")
            out_path = os.path.join(tmpdir, "skels.json")

            np.save(seg_path, seg_subset)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            proc = subprocess.run(
                [sys.executable, "-c", _KIMIMARO_WORKER_CODE, seg_path, cfg_path, out_path],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                err_tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
                raise RuntimeError(f"Parallel Kimimaro worker failed (exit {proc.returncode}): {err_tail}")

            with open(out_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            return {int(k): v for k, v in raw_map.items()}

    import kimimaro

    skels = kimimaro.skeletonize(
        seg_subset,
        teasar_params={
            "scale": cfg["scale"],
            "const": cfg["const"],
            "pdrf_scale": cfg["pdrf_scale"],
            "pdrf_exponent": cfg["pdrf_exponent"],
            "soma_acceptance_threshold": cfg["pdrf_soma_acceptance_threshold"],
            "soma_detection_threshold": cfg["pdrf_soma_detection_threshold"],
            "soma_invalidation_const": cfg["pdrf_soma_invalidation_const"],
            "soma_invalidation_scale": cfg["pdrf_soma_invalidation_scale"],
            "max_paths": cfg["max_paths"],
        },
        dust_threshold=cfg["dust_threshold"],
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
        fix_avocados=False,
        progress=False,
        parallel=parallel_workers,
        parallel_chunk_size=1,
    )
    return {int(rid): skel.to_swc() for rid, skel in skels.items()}


def _postprocess_kimimaro_swc(
    swc_text: str,
    root_id: int,
    bbox_offset_vox: np.ndarray,
    mip_resolution_nm: np.ndarray,
    catmaid_scale: np.ndarray,
    smooth_window: int = 2,
    resample_to_nm: float = 128.0,
) -> navis.TreeNeuron:
    """Replicate ZlaticLab postprocess_skeleton: voxel->nm scale, heal, smooth, resample, convert to CATMAID units."""
    raw_neuron = navis.read_swc(swc_text, id=int(root_id))
    if len(raw_neuron.nodes) == 0:
        raise ValueError(f"Kimimaro produced 0 skeleton nodes for CAVE root {root_id}.")

    raw_neuron.nodes[["x", "y", "z"]] = (
        raw_neuron.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) + bbox_offset_vox
    ) * mip_resolution_nm

    healed = navis.heal_skeleton(raw_neuron)

    if smooth_window and int(smooth_window) > 1 and len(healed.nodes) > int(smooth_window):
        smoothed = navis.smooth_skeleton(healed, window=int(smooth_window))
    else:
        smoothed = healed

    if resample_to_nm and float(resample_to_nm) > 0 and len(smoothed.nodes) > 2:
        resampled = smoothed.resample(resample_to=float(resample_to_nm))
    else:
        resampled = smoothed

    resampled.nodes[["x", "y", "z"]] = (
        resampled.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) / catmaid_scale
    )
    resampled.id = int(root_id)
    resampled.name = f"cave_{int(root_id)}"
    return resampled


def skeletonize_cave_roots_kimimaro(
    cave_client: CAVEclient,
    root_ids: list[int],
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
    mip: int = 3,
    resample_to_nm: float = 128.0,
    smooth_window: int = 2,
    parallel: int = 4,
    progress_callback=None,
) -> dict[int, navis.TreeNeuron]:
    """Batch-skeletonize CAVE roots using ZlaticLab's Kimimaro TEASAR + navis pipeline."""
    try:
        import kimimaro  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "kimimaro is required for CAVE skeletonization. Install it with: pip install kimimaro"
        ) from exc

    clean_ids = [int(r) for r in root_ids if int(r) != 0]
    if not clean_ids:
        raise ValueError("No valid (non-zero) CAVE segment IDs provided.")

    cfg = DEFAULT_KIMIMARO_CONFIG.copy()
    cfg["mip"] = int(mip)
    cfg["resample_to"] = float(resample_to_nm)
    cfg["smooth_window"] = int(smooth_window)
    cfg["parallel"] = max(1, int(parallel))

    if progress_callback:
        progress_callback(f"Opening CAVE segmentation volume at mip={cfg['mip']}...")

    vol_seg = _retry_call(
        lambda: cave_client.info.segmentation_cloudvolume(agglomerate=True, mip=cfg["mip"]),
        stage=f"CAVE segmentation_cloudvolume(mip={cfg['mip']})",
        progress_callback=progress_callback,
    )
    vol_seg.fill_missing = True
    vol_seg.progress = False

    if progress_callback:
        progress_callback(f"Computing tight bounding box for {len(clean_ids)} segment(s)...")

    dl_bbox = _compute_tight_mip_bbox(
        cave_client,
        clean_ids,
        vol_seg,
        pad_chunks=1,
        progress_callback=progress_callback,
    )
    box_vox = np.asarray(dl_bbox.size3(), dtype=np.int64)

    if progress_callback:
        progress_callback(
            f"Downloading segmentation at mip={cfg['mip']} ({box_vox[0]}x{box_vox[1]}x{box_vox[2]} voxels)..."
        )

    seg_subset = _retry_call(
        lambda: vol_seg.download(bbox=dl_bbox, agglomerate=False, segids=clean_ids),
        stage=f"CloudVolume download(mip={cfg['mip']})",
        progress_callback=progress_callback,
    )
    if seg_subset.ndim == 4:
        seg_subset = seg_subset[:, :, :, 0]

    if progress_callback:
        progress_callback(
            f"Running Kimimaro TEASAR skeletonization at mip={cfg['mip']} ({cfg['parallel']} worker(s))..."
        )

    swc_by_id = _run_kimimaro_parallel_swc(seg_subset, cfg)

    bbox_offset_vox = np.asarray(dl_bbox.minpt, dtype=np.float64)
    mip_resolution_nm = np.asarray(vol_seg.resolution, dtype=np.float64)
    catmaid_scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    out: dict[int, navis.TreeNeuron] = {}
    for rid in clean_ids:
        if rid not in swc_by_id:
            raise ValueError(
                f"Kimimaro found no voxels for CAVE root {rid} at mip={cfg['mip']}. "
                f"Try lowering MIP (e.g. mip=3) or verify the segment ID is current."
            )
        if progress_callback:
            progress_callback(f"Healing, smoothing & resampling skeleton for {rid}...")
        out[rid] = _postprocess_kimimaro_swc(
            swc_by_id[rid],
            root_id=rid,
            bbox_offset_vox=bbox_offset_vox,
            mip_resolution_nm=mip_resolution_nm,
            catmaid_scale=catmaid_scale,
            smooth_window=cfg["smooth_window"],
            resample_to_nm=cfg["resample_to"],
        )
    return out


def skeletonize_cave_root(
    cave_client: CAVEclient,
    root_id: int,
    catmaid_resolution_nm: tuple[float, float, float],
    catmaid_coord_unit: str = "nm",
    mip: int = 3,
    resample_to_nm: float = 128.0,
    smooth_window: int = 2,
    parallel: int = 4,
) -> navis.TreeNeuron:
    """Skeletonize a single CAVE root using ZlaticLab's Kimimaro + navis pipeline."""
    res = skeletonize_cave_roots_kimimaro(
        cave_client=cave_client,
        root_ids=[int(root_id)],
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
        mip=mip,
        resample_to_nm=resample_to_nm,
        smooth_window=smooth_window,
        parallel=parallel,
    )
    return res[int(root_id)]


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
        for ann in ann_list:
            try:
                found = _retry_call(
                    lambda a=ann: pymaid.get_skids_by_annotation(a, remote_instance=rm),
                    stage=f"CATMAID annotation '{ann}'",
                    progress_callback=progress_callback,
                )
                skids.update(int(s) for s in found)
            except Exception:
                continue
    elif local_boxes_nm:
        skids.update(
            _retry_call(
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

    neurons = _retry_call(
        lambda: pymaid.get_neurons(list(skids), remote_instance=rm),
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
        internal_records.append(internal[["skeleton_id", "node_id", "parent_id", "x", "y", "z"]])

    internal_df = pd.concat(internal_records, ignore_index=True) if internal_records else pd.DataFrame()
    return trees_by_skid, internal_df


def _point_to_segments_min_dist(pts_nm: np.ndarray, existing_df: pd.DataFrame) -> np.ndarray:
    """Compute minimum 3D distance (nm) from each point in pts_nm to existing_df's parent->child segments."""
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
    base_neuron_name: str = "",
    search_annotations: str = "",
    cut_buffer_nm: float = 400.0,
    slice_step: int = 4,
    mip: int = 3,
    resample_to_nm: float = 128.0,
    parallel: int = 4,
    min_fragment_nodes: int = 3,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """Skeletonize CAVE roots with Kimimaro, pair with existing CATMAID nodes, cut & decimate."""
    raw_unique = list(dict.fromkeys(int(r) for r in cave_root_ids))
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    if progress_callback:
        progress_callback("Verifying latest CAVE root IDs...")

    latest_roots: list[int] = []
    for rid in raw_unique:
        is_latest = _retry_call(
            lambda r=rid: cave_client.chunkedgraph.is_latest_roots([r])[0],
            stage=f"CAVE is_latest_roots({rid})",
            progress_callback=progress_callback,
        )
        if not is_latest:
            latest = _retry_call(
                lambda r=rid: cave_client.chunkedgraph.get_latest_roots(r),
                stage=f"CAVE get_latest_roots({rid})",
                progress_callback=progress_callback,
            )
            rid = int(latest[0])
        if rid not in latest_roots:
            latest_roots.append(rid)

    cave_skeletons = skeletonize_cave_roots_kimimaro(
        cave_client=cave_client,
        root_ids=latest_roots,
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
        mip=mip,
        resample_to_nm=resample_to_nm,
        parallel=parallel,
        progress_callback=progress_callback,
    )

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
        progress_callback("Loading existing CATMAID skeletons from search annotations...")

    trees_by_skid, internal_df = load_existing_catmaid_trees(
        rm,
        search_annotations=search_annotations,
        local_boxes_nm=local_boxes_nm,
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
        progress_callback=progress_callback,
    )

    root_to_skids: dict[int, list[tuple[int, int]]] = {}
    if not internal_df.empty:
        mapped_cm = _retry_call(
            lambda: map_nodes_to_cave(
                internal_df,
                cave_client,
                catmaid_resolution_nm=catmaid_resolution_nm,
                cave_resolution_nm=cave_resolution_nm,
                catmaid_coord_unit=catmaid_coord_unit,
            ),
            stage="CAVE map_nodes_to_cave",
            progress_callback=progress_callback,
        )
        for (r_id, skid), grp in mapped_cm.groupby(["root_id", "skeleton_id"]):
            root_to_skids.setdefault(int(r_id), []).append((int(skid), int(len(grp))))

    user_base = (base_neuron_name or "").strip()
    plan_rows = []
    bundle_items = {}

    for root_idx, (rid, nrn) in enumerate(cave_skeletons.items(), start=1):
        total_nodes = int(len(nrn.nodes))
        cave_pts_nm = nrn.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale

        matched_pairs = sorted(root_to_skids.get(rid, []), key=lambda x: -x[1])
        matched_skids = [skid for skid, _ in matched_pairs if skid in trees_by_skid]

        if not matched_skids:
            match_type = "0-Match (New)"
            master_skid = None
            keep_mask = np.ones(total_nodes, dtype=bool)
            combined_existing = pd.DataFrame()
        else:
            master_skid = matched_skids[0]
            match_type = "1-to-1 Match" if len(matched_skids) == 1 else f"1-to-Many ({len(matched_skids)} SKIDs)"
            combined_existing = pd.concat(
                [trees_by_skid[s]["existing_nodes"] for s in matched_skids],
                ignore_index=True,
            )
            min_dists = _point_to_segments_min_dist(cave_pts_nm, combined_existing)
            keep_mask = min_dists > float(cut_buffer_nm)

        if user_base:
            base_name = user_base.replace("{n}", str(root_idx)) if "{n}" in user_base else user_base
        elif master_skid is not None:
            base_name = trees_by_skid[master_skid]["name"]
        else:
            base_name = f"cave_{rid}"

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
            frag_name = base_name if n_frags == 1 else f"{base_name}_{idx_f}"

            anchor_info = None
            if not combined_existing.empty:
                sub_pts_nm = sub_df[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
                m_pts_nm = combined_existing[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
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

        frag_names_preview = ", ".join(f["name"] for f in fragment_specs) if fragment_specs else ""
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
    pair = _retry_call(
        lambda: pymaid.get_neurons([int(master_skid), int(child_skid)], remote_instance=rm),
        stage="CATMAID get_neurons(stitch)",
    )
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
        _retry_call(
            lambda: pymaid.join_skeletons(
                int(child_skid),
                int(master_skid),
                from_node=child_nid,
                to_node=master_nid,
                remote_instance=rm,
            ),
            stage="CATMAID join_skeletons",
        )
    else:
        url = rm.make_url(rm.project_id, "skeleton", "join")
        _retry_call(
            lambda: rm.fetch(
                url,
                post={
                    "from_id": int(child_skid),
                    "to_id": int(master_skid),
                    "from_node_id": child_nid,
                    "to_node_id": master_nid,
                },
            ),
            stage="CATMAID skeleton/join",
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
    """Upload approved non-overlapping CAVE fragments to CATMAID without node tags, stitch & annotate."""
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

        edited_names = [x.strip() for x in str(row.get("fragment_names") or "").split(",") if x.strip()]
        master_skid = item["master_skid"]
        created_skids = []
        created_names = []
        nodes_added = 0
        notes = []

        try:
            for f_idx, frag in enumerate(item["fragments"]):
                frag_name = (
                    edited_names[f_idx]
                    if f_idx < len(edited_names)
                    else (edited_names[0] if len(edited_names) == 1 and len(item["fragments"]) == 1 else frag["name"])
                )
                sub_df = frag["nodes_df"]
                nrn = navis.TreeNeuron(sub_df)
                nrn.name = frag_name

                up_res = _retry_call(
                    lambda n=nrn: pymaid.upload_neuron(
                        n,
                        remote_instance=rm,
                        import_tags=False,
                        import_annotations=False,
                        import_connectors=False,
                    ),
                    stage=f"CATMAID upload_neuron({frag_name})",
                    progress_callback=progress_callback,
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
                        notes.append(f"Stitched {frag_name} into SKID {master_skid} ({dist_nm:.0f} nm)")
                    except Exception as join_exc:
                        created_skids.append(new_skid)
                        created_names.append(frag_name)
                        notes.append(f"Kept {frag_name} (SKID {new_skid}) separate: {join_exc}")
                else:
                    created_skids.append(new_skid)
                    created_names.append(frag_name)

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
                _retry_call(
                    lambda: pymaid.add_annotations(final_skids, ann_list, remote_instance=rm),
                    stage="CATMAID add_annotations",
                    progress_callback=progress_callback,
                )

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