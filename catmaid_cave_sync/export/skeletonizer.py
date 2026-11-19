"""Fast sparse L2-cluster CloudVolume download + Kimimaro TEASAR + navis skeletonization."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import navis
import numpy as np
from caveclient import CAVEclient

from .retry import retry_call

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
    "dust_threshold": 50,
    "parallel": 4,
    "smooth_window": 2,
    "resample_to": 128.0,
}


def _report(progress_callback, msg: str, pct: float | None = None):
    if not progress_callback:
        return
    try:
        progress_callback(msg, pct)
    except TypeError:
        progress_callback(msg)


def _merge_chunk_coords_to_boxes(
    pts_nm: np.ndarray,
    full_min: np.ndarray,
    full_max: np.ndarray,
    mip_res: np.ndarray,
    chunk_sz: np.ndarray,
    pad_chunks: int = 1,
    max_sub_boxes: int = 32,
):
    """Group sparse 3D points into tight, non-overlapping chunk-aligned Bboxes so empty space is never downloaded."""
    from cloudvolume.lib import Bbox

    raw_vox = np.floor(pts_nm / mip_res).astype(np.int64)
    rel_chunks = np.floor((raw_vox - full_min) / chunk_sz).astype(np.int64)
    unique_chunks = np.unique(rel_chunks, axis=0)

    max_chunk_idx = np.maximum(np.ceil((full_max - full_min) / chunk_sz).astype(np.int64) - 1, 0)
    shifts = np.array(
        [[dx, dy, dz] for dx in range(-pad_chunks, pad_chunks + 1)
         for dy in range(-pad_chunks, pad_chunks + 1)
         for dz in range(-pad_chunks, pad_chunks + 1)],
        dtype=np.int64,
    )
    padded = (unique_chunks[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    padded = np.clip(padded, 0, max_chunk_idx)
    cells = np.unique(padded, axis=0)

    coarsen = 1
    while True:
        c_cells = np.unique(cells // coarsen, axis=0)
        c_cells = c_cells[np.lexsort((c_cells[:, 0], c_cells[:, 1], c_cells[:, 2]))]

        x_runs = []
        for ix, iy, iz in c_cells:
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

        if len(xyz_runs) <= max_sub_boxes:
            break
        coarsen *= 2

    sub_boxes = []
    for x0, x1, y0, y1, z0, z1 in xyz_runs:
        c_min = np.array([x0, y0, z0], dtype=np.int64) * coarsen
        c_max = (np.array([x1, y1, z1], dtype=np.int64) + 1) * coarsen
        min_vox = np.maximum(full_min + c_min * chunk_sz, full_min)
        max_vox = np.minimum(full_min + c_max * chunk_sz, full_max)
        max_vox = np.maximum(max_vox, min_vox + 1)
        sub_boxes.append(Bbox(min_vox.tolist(), max_vox.tolist()))

    all_min = np.min([b.minpt for b in sub_boxes], axis=0).tolist()
    all_max = np.max([b.maxpt for b in sub_boxes], axis=0).tolist()
    return Bbox(all_min, all_max), sub_boxes


def _compute_root_sparse_boxes(
    cave_client: CAVEclient,
    root_id: int,
    vol_seg,
    pad_chunks: int = 1,
    progress_callback=None,
):
    """Return (union_bbox, sub_boxes) covering ONLY the occupied L2 chunks of a single root_id."""
    from cloudvolume.lib import Bbox

    full_bounds = vol_seg.bounds
    full_min = np.asarray(full_bounds.minpt, dtype=np.int64)
    full_max = np.asarray(full_bounds.maxpt, dtype=np.int64)
    mip_res = np.asarray(vol_seg.resolution, dtype=np.float64)
    chunk_sz = np.maximum(np.asarray(getattr(vol_seg, "chunk_size", [64, 64, 64])[:3], dtype=np.int64), 1)

    pts_nm = []
    try:
        l2_ids = retry_call(
            lambda r=int(root_id): cave_client.chunkedgraph.get_leaves(r, stop_layer=2),
            stage=f"CAVE get_leaves({root_id})",
            progress_callback=progress_callback,
        )
        if len(l2_ids) > 0 and getattr(cave_client, "l2cache", None) is not None:
            id_list = [int(x) for x in l2_ids]
            batches = [id_list[i : i + 5000] for i in range(0, len(id_list), 5000)]

            def _fetch_l2_batch(b):
                return retry_call(
                    lambda: cave_client.l2cache.get_l2data(b, attributes=["rep_coord_nm"]),
                    stage="CAVE l2cache",
                    progress_callback=progress_callback,
                )

            with ThreadPoolExecutor(max_workers=min(4, max(1, len(batches)))) as ex:
                for info in ex.map(_fetch_l2_batch, batches):
                    for v in info.values():
                        pt = v.get("rep_coord_nm") if isinstance(v, dict) else None
                        if pt is not None and len(pt) == 3:
                            pts_nm.append(pt)
    except Exception:
        pts_nm = []

    if pts_nm:
        return _merge_chunk_coords_to_boxes(
            np.asarray(pts_nm, dtype=np.float64),
            full_min=full_min,
            full_max=full_max,
            mip_res=mip_res,
            chunk_sz=chunk_sz,
            pad_chunks=pad_chunks,
        )

    try:
        mesh = retry_call(
            lambda r=int(root_id): vol_seg.mesh.get(r, deduplicate_chunk_boundaries=False),
            stage=f"CAVE mesh({root_id})",
            progress_callback=progress_callback,
        )
        m_obj = mesh[int(root_id)] if isinstance(mesh, dict) else mesh
        verts = np.asarray(m_obj.vertices, dtype=np.float64)
        if len(verts) > 0:
            raw_min = np.floor(verts.min(axis=0) / mip_res).astype(np.int64)
            raw_max = np.ceil(verts.max(axis=0) / mip_res).astype(np.int64)
            rel_min = np.floor((raw_min - full_min) / chunk_sz).astype(np.int64) - pad_chunks
            rel_max = np.ceil((raw_max - full_min) / chunk_sz).astype(np.int64) + pad_chunks
            min_vox = np.maximum(full_min + rel_min * chunk_sz, full_min)
            max_vox = np.minimum(full_min + rel_max * chunk_sz, full_max)
            max_vox = np.maximum(max_vox, min_vox + 1)
            box = Bbox(min_vox.tolist(), max_vox.tolist())
            return box, [box]
    except Exception:
        pass

    return full_bounds, [full_bounds]


def _download_sparse_root_volume(
    vol_seg,
    root_id: int,
    union_bbox,
    sub_boxes: list,
    progress_callback=None,
) -> np.ndarray:
    """Download only the occupied sub-boxes in parallel and assemble into a local uint64 cutout."""
    box_shape = tuple(int(x) for x in union_bbox.size3())
    seg_arr = np.zeros(box_shape, dtype=np.uint64)
    u_min = np.asarray(union_bbox.minpt, dtype=np.int64)

    def _dl_one(sb):
        cut = retry_call(
            lambda: vol_seg.download(bbox=sb, agglomerate=False, segids=[int(root_id)]),
            stage=f"CloudVolume download({root_id})",
            progress_callback=progress_callback,
        )
        if cut.ndim == 4:
            cut = cut[:, :, :, 0]
        return sb, cut

    workers = min(8, max(1, len(sub_boxes)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sb, cut in ex.map(_dl_one, sub_boxes):
            off = np.asarray(sb.minpt, dtype=np.int64) - u_min
            end = off + np.asarray(cut.shape, dtype=np.int64)
            seg_arr[off[0]:end[0], off[1]:end[1], off[2]:end[2]] = cut

    return seg_arr


def _run_kimimaro_swc(seg_subset: np.ndarray, cfg: dict) -> tuple[str | None, np.ndarray]:
    """Crop to tight foreground bbox, cast to uint8 (8x smaller), and run Kimimaro in-process."""
    import kimimaro

    nz = np.nonzero(seg_subset)
    if len(nz[0]) == 0:
        return None, np.zeros(3, dtype=np.float64)

    min_idx = np.array([int(nz[0].min()), int(nz[1].min()), int(nz[2].min())], dtype=np.int64)
    max_idx = np.array([int(nz[0].max()) + 1, int(nz[1].max()) + 1, int(nz[2].max()) + 1], dtype=np.int64)
    cropped_mask = (
        seg_subset[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] != 0
    ).astype(np.uint8)

    skels = kimimaro.skeletonize(
        cropped_mask,
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
        dust_threshold=int(cfg["dust_threshold"]),
        fix_branching=True,
        fix_borders=False,
        fill_holes=False,
        fix_avocados=False,
        progress=False,
        parallel=1,
        parallel_chunk_size=1,
    )
    if 1 not in skels:
        return None, min_idx.astype(np.float64)
    return skels[1].to_swc(), min_idx.astype(np.float64)


def _postprocess_kimimaro_swc(
    swc_text: str,
    root_id: int,
    bbox_offset_vox: np.ndarray,
    mip_resolution_nm: np.ndarray,
    catmaid_scale: np.ndarray,
    smooth_window: int = 2,
    resample_to_nm: float = 128.0,
) -> navis.TreeNeuron:
    """Scale voxel->nm, prune tiny dust fragments before O(K^2) healing, smooth, resample, convert to CATMAID units."""
    raw_neuron = navis.read_swc(swc_text, id=int(root_id))
    if len(raw_neuron.nodes) == 0:
        raise ValueError(f"Kimimaro produced 0 skeleton nodes for CAVE root {root_id}.")

    if raw_neuron.n_trees > 1:
        subtrees = navis.break_fragments(raw_neuron)
        big = [t for t in subtrees if len(t.nodes) >= 3]
        if big:
            raw_neuron = navis.combine_neurons(big)

    raw_neuron.nodes[["x", "y", "z"]] = (
        raw_neuron.nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) + bbox_offset_vox
    ) * mip_resolution_nm

    healed = navis.heal_skeleton(raw_neuron) if raw_neuron.n_trees > 1 else raw_neuron

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
    pct_range: tuple[float, float] = (8.0, 65.0),
) -> dict[int, navis.TreeNeuron]:
    """Skeletonize each CAVE root independently using sparse L2-chunk downloads + Kimimaro TEASAR + navis."""
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

    p_start, p_end = float(pct_range[0]), float(pct_range[1])
    _report(progress_callback, f"Opening CAVE segmentation volume at mip={cfg['mip']}...", p_start)

    vol_seg = retry_call(
        lambda: cave_client.info.segmentation_cloudvolume(agglomerate=True, mip=cfg["mip"]),
        stage=f"CAVE segmentation_cloudvolume(mip={cfg['mip']})",
        progress_callback=progress_callback,
    )
    vol_seg.fill_missing = True
    vol_seg.progress = False

    mip_resolution_nm = np.asarray(vol_seg.resolution, dtype=np.float64)
    catmaid_scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    n_total = len(clean_ids)
    step_span = (p_end - p_start) / max(1, n_total)
    prog_lock = threading.Lock()
    done_cnt = 0

    def _process_one_root(idx: int, rid: int) -> tuple[int, navis.TreeNeuron]:
        nonlocal done_cnt
        base_pct = p_start + idx * step_span
        if n_total == 1:
            _report(progress_callback, f"[{rid}] Computing sparse L2 chunk boxes...", base_pct + 0.1 * step_span)
        union_bbox, sub_boxes = _compute_root_sparse_boxes(
            cave_client,
            rid,
            vol_seg,
            pad_chunks=1,
            progress_callback=progress_callback,
        )
        if n_total == 1:
            _report(
                progress_callback,
                f"[{rid}] Downloading {len(sub_boxes)} sparse sub-box(es) at mip={cfg['mip']}...",
                base_pct + 0.3 * step_span,
            )
        seg_subset = _download_sparse_root_volume(
            vol_seg,
            rid,
            union_bbox,
            sub_boxes,
            progress_callback=progress_callback,
        )
        if n_total == 1:
            _report(progress_callback, f"[{rid}] Running Kimimaro TEASAR & navis smoothing...", base_pct + 0.7 * step_span)
        swc_text, crop_offset_vox = _run_kimimaro_swc(seg_subset, cfg)
        if swc_text is None:
            raise ValueError(
                f"Kimimaro found no voxels for CAVE root {rid} at mip={cfg['mip']}. "
                f"Try lowering MIP (e.g. mip=3) or verify the segment ID is current."
            )
        nrn = _postprocess_kimimaro_swc(
            swc_text,
            root_id=rid,
            bbox_offset_vox=np.asarray(union_bbox.minpt, dtype=np.float64) + crop_offset_vox,
            mip_resolution_nm=mip_resolution_nm,
            catmaid_scale=catmaid_scale,
            smooth_window=cfg["smooth_window"],
            resample_to_nm=cfg["resample_to"],
        )
        with prog_lock:
            done_cnt += 1
            cur_pct = p_start + (done_cnt / n_total) * (p_end - p_start)
            _report(progress_callback, f"Skeletonized {done_cnt}/{n_total} CAVE segment(s)...", cur_pct)
        return rid, nrn

    out: dict[int, navis.TreeNeuron] = {}
    if n_total == 1:
        rid = clean_ids[0]
        _, out[rid] = _process_one_root(0, rid)
    else:
        _report(progress_callback, f"Skeletonizing {n_total} segments across {cfg['parallel']} worker(s) (mip={cfg['mip']})...", p_start + 2.0)
        root_workers = min(cfg["parallel"], n_total)
        with ThreadPoolExecutor(max_workers=root_workers) as ex:
            futures = {ex.submit(_process_one_root, i, rid): rid for i, rid in enumerate(clean_ids)}
            for fut in as_completed(futures):
                rid, nrn = fut.result()
                out[rid] = nrn

    return {rid: out[rid] for rid in clean_ids}


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
    """Skeletonize a single CAVE root using sparse L2-chunk Kimimaro + navis."""
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