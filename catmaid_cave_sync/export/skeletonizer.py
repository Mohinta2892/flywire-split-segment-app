"""Skeletor (mesh wavefront/TEASAR) + navis skeletonization of CAVE segments with radius-aware twig pruning."""
import contextlib
import inspect
import io
import warnings
from typing import Callable

import navis
import numpy as np
import pandas as pd
from caveclient import CAVEclient

try:
    import skeletor as sk
    import trimesh
except ImportError as _imp_err:
    sk = None
    trimesh = None
    _SKELETOR_IMPORT_ERR = _imp_err
else:
    _SKELETOR_IMPORT_ERR = None

from .retry import retry_call

DEFAULT_SKELETOR_CONFIG: dict = {
    "method": "wavefront",
    "waves": 2,
    "step_size": 1,
    "exact_step_nm": 150.0,
    "inv_dist": 1000.0,
    "remove_disconnected": 5,
    "min_twig_nm": 500.0,
    "twig_radius_factor": 2.2,
    "resample_to_nm": 160.0,
    "max_heal_dist_nm": 2500.0,
    "smooth_window": 5,
}


def _report(progress_callback: Callable | None, message: str, pct: float | None = None) -> None:
    if not progress_callback:
        return
    try:
        if pct is not None:
            progress_callback(message, float(pct))
        else:
            progress_callback(message)
    except TypeError:
        progress_callback(message)


def _fetch_cave_trimesh(cv, root_id: int):
    """Download a CAVE segment mesh in nm, weld chunk seams, and return a clean trimesh.Trimesh."""
    mesh_sig = inspect.signature(cv.mesh.get)
    kw = {"deduplicate": False} if "deduplicate" in mesh_sig.parameters else {}

    # Suppress CloudVolume's noisy 'deduplication not currently supported...' stdout/stderr warning
    sink = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        warnings.simplefilter("ignore")
        mesh_obj = cv.mesh.get(int(root_id), **kw)

    raw = mesh_obj[int(root_id)] if isinstance(mesh_obj, dict) else mesh_obj
    verts = np.asarray(raw.vertices, dtype=np.float64)
    faces = np.asarray(raw.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError(f"CAVE root {root_id} returned an empty mesh (0 vertices or faces).")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Weld duplicate vertices along Graphene octree chunk boundaries so the neuron is one continuous mesh
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def _make_clean_tree_neuron(df: pd.DataFrame, rid: int, name: str) -> navis.TreeNeuron:
    """Construct a navis.TreeNeuron from a deep-copied DataFrame (avoids pandas SettingWithCopyWarning)."""
    clean = df.copy(deep=True).reset_index(drop=True)
    clean["node_id"] = clean["node_id"].astype(np.int64)
    clean["parent_id"] = clean["parent_id"].fillna(-1).astype(np.int64)
    for col in ("x", "y", "z"):
        clean[col] = clean[col].astype(np.float64)
    if "radius" in clean.columns:
        clean["radius"] = np.nan_to_num(clean["radius"].to_numpy(dtype=np.float64), nan=10.0)
    else:
        clean["radius"] = 10.0
    clean["type"] = "slab"
    cols = ["node_id", "parent_id", "x", "y", "z", "radius", "type"]
    return navis.TreeNeuron(clean[cols].copy(deep=True), id=int(rid), name=name)


def _skeletor_to_tree_neuron(skel_obj, rid: int) -> navis.TreeNeuron:
    """Convert a skeletor.Skeleton into a 1-indexed navis.TreeNeuron preserving mesh radius."""
    swc = skel_obj.swc.copy(deep=True)
    if len(swc) == 0:
        raise RuntimeError(f"Skeletor produced 0 nodes for CAVE root {rid}.")

    if int(swc["node_id"].min()) == 0:
        swc["node_id"] = swc["node_id"].astype(np.int64) + 1
        has_parent = swc["parent_id"] >= 0
        swc.loc[has_parent, "parent_id"] = swc.loc[has_parent, "parent_id"].astype(np.int64) + 1
        swc.loc[~has_parent, "parent_id"] = -1

    return _make_clean_tree_neuron(swc, rid=int(rid), name=f"cave_{rid}")


def _prune_spurious_twigs(
    nrn: navis.TreeNeuron,
    min_twig_nm: float = 500.0,
    radius_factor: float = 2.2,
    max_passes: int = 25,
) -> navis.TreeNeuron:
    """Iteratively prune short terminal surface hairs and root spurs until no branch < threshold remains."""
    if min_twig_nm <= 0 and radius_factor <= 0:
        return nrn

    nodes = nrn.nodes.copy(deep=True)
    for _ in range(max(1, int(max_passes))):
        n = len(nodes)
        if n < 4:
            break

        node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
        parent_ids = nodes["parent_id"].fillna(-1).to_numpy(dtype=np.int64)
        coords = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if "radius" in nodes.columns:
            radii = np.nan_to_num(nodes["radius"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            radii = np.zeros(n, dtype=np.float64)

        id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
        parent_idx = np.array([id_to_idx.get(int(pid), -1) for pid in parent_ids], dtype=np.int64)
        children: list[list[int]] = [[] for _ in range(n)]
        for i, p in enumerate(parent_idx):
            if p >= 0:
                children[p].append(i)

        twigs_by_bp: dict[int, list[tuple[float, list[int]]]] = {}
        for leaf in range(n):
            if len(children[leaf]) != 0:
                continue
            path = [leaf]
            length = 0.0
            curr = leaf
            bp = -1
            while True:
                p = int(parent_idx[curr])
                if p < 0:
                    break
                length += float(np.linalg.norm(coords[curr] - coords[p]))
                if len(children[p]) >= 2:
                    bp = p
                    break
                path.append(p)
                curr = p

            if bp >= 0:
                thresh = max(float(min_twig_nm), float(radius_factor) * float(radii[bp]))
                if length < thresh:
                    twigs_by_bp.setdefault(bp, []).append((length, path))

        to_delete: set[int] = set()
        for bp, twig_list in twigs_by_bp.items():
            # When all children of a fork are short terminal twigs:
            # - If bp is the root, keep the longest child.
            # - If bp has a parent shaft, check total stem + longest-tip length:
            #   delete all children if the entire forked spur is < thresh (so bp collapses
            #   and is pruned on the next pass), otherwise keep the longest tip.
            if len(twig_list) >= len(children[bp]):
                twig_list.sort(key=lambda x: x[0], reverse=True)
                keep_longest = parent_idx[bp] < 0
                if not keep_longest:
                    stem_len = 0.0
                    curr = bp
                    while True:
                        p = int(parent_idx[curr])
                        if p < 0:
                            break
                        stem_len += float(np.linalg.norm(coords[curr] - coords[p]))
                        if len(children[p]) >= 2:
                            break
                        curr = p
                    bp_thresh = max(float(min_twig_nm), float(radius_factor) * float(radii[bp]))
                    if (stem_len + twig_list[0][0]) >= bp_thresh:
                        keep_longest = True
                if keep_longest:
                    twig_list = twig_list[1:]
            for _, path in twig_list:
                to_delete.update(path)

        # Also prune short root-side spurs before the first branch point
        new_roots: set[int] = set()
        for r in range(n):
            if parent_idx[r] >= 0 or r in to_delete or len(children[r]) != 1:
                continue
            path = [r]
            length = 0.0
            curr = r
            bp = -1
            while len(children[curr]) == 1:
                nxt = children[curr][0]
                if nxt in to_delete:
                    break
                length += float(np.linalg.norm(coords[nxt] - coords[curr]))
                if len(children[nxt]) >= 2:
                    bp = nxt
                    break
                path.append(nxt)
                curr = nxt
            if bp >= 0:
                thresh = max(float(min_twig_nm), float(radius_factor) * float(radii[bp]))
                if length < thresh:
                    to_delete.update(path)
                    new_roots.add(int(node_ids[bp]))

        if not to_delete:
            break

        keep_mask = np.array([i not in to_delete for i in range(n)], dtype=bool)
        if keep_mask.sum() < 2:
            break
        nodes = nodes.loc[keep_mask].copy(deep=True)
        if new_roots:
            nodes.loc[nodes["node_id"].isin(new_roots), "parent_id"] = -1

    return _make_clean_tree_neuron(nodes, rid=int(nrn.id), name=nrn.name)

def skeletonize_cave_roots_skeletor(
    cave_client: CAVEclient,
    root_ids: list[int],
    catmaid_resolution_nm: tuple[float, float, float] = (8.0, 8.0, 16.0),
    catmaid_coord_unit: str = "nm",
    skeletor_method: str | None = None,
    min_twig_nm: float | None = None,
    resample_to_nm: float | None = None,
    max_heal_dist_nm: float | None = None,
    skeletor_config: dict | None = None,
    progress_callback=None,
    pct_range: tuple[float, float] = (8.0, 62.0),
) -> dict[int, navis.TreeNeuron]:
    """Skeletonize CAVE segments from their 3D meshes via skeletor and refine with navis."""
    if sk is None or trimesh is None:
        raise RuntimeError(
            f"skeletor / trimesh is not installed ({_SKELETOR_IMPORT_ERR}). "
            "Install with: pip install skeletor trimesh"
        )
    if not root_ids:
        return {}

    cfg = {**DEFAULT_SKELETOR_CONFIG, **(skeletor_config or {})}
    method = str(skeletor_method or cfg.get("method", "wavefront")).strip().lower()
    eff_min_twig = float(cfg["min_twig_nm"] if min_twig_nm is None else min_twig_nm)
    radius_factor = float(cfg.get("twig_radius_factor", 2.2)) if eff_min_twig > 0 else 0.0
    eff_resample = float(cfg["resample_to_nm"] if resample_to_nm is None else resample_to_nm)
    eff_heal_dist = float(cfg["max_heal_dist_nm"] if max_heal_dist_nm is None else max_heal_dist_nm)
    smooth_window = int(cfg.get("smooth_window", 5))
    remove_disc = int(cfg.get("remove_disconnected", 5))

    cv_sig = inspect.signature(cave_client.info.segmentation_cloudvolume)
    cv_kw = {"progress": False} if "progress" in cv_sig.parameters else {}
    cv = retry_call(
        lambda: cave_client.info.segmentation_cloudvolume(**cv_kw),
        stage="CAVE segmentation_cloudvolume",
        progress_callback=progress_callback,
    )

    cm_scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    out: dict[int, navis.TreeNeuron] = {}
    p_start, p_end = float(pct_range[0]), float(pct_range[1])
    n_total = max(1, len(root_ids))

    for idx, rid in enumerate(root_ids):
        rid = int(rid)
        base_pct = p_start + (idx / n_total) * (p_end - p_start)
        span_pct = (p_end - p_start) / n_total

        _report(progress_callback, f"[{idx + 1}/{len(root_ids)}] Downloading 3D mesh for CAVE root {rid}...", base_pct)
        mesh = retry_call(
            lambda r=rid: _fetch_cave_trimesh(cv, r),
            stage=f"CAVE mesh({rid})",
            progress_callback=progress_callback,
        )

        _report(
            progress_callback,
            f"[{idx + 1}/{len(root_ids)}] Cleaning mesh ({len(mesh.vertices):,} vertices) for {rid}...",
            base_pct + 0.30 * span_pct,
        )
        mesh = sk.pre.fix_mesh(mesh, remove_disconnected=remove_disc, inplace=False)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise RuntimeError(f"Mesh for CAVE root {rid} had no valid faces after cleanup.")

        _report(
            progress_callback,
            f"[{idx + 1}/{len(root_ids)}] Skeletonizing {rid} via skeletor ({method})...",
            base_pct + 0.55 * span_pct,
        )
        if method == "wavefront_exact":
            skel_obj = sk.skeletonize.by_wavefront_exact(
                mesh,
                step_size=float(cfg.get("exact_step_nm", 150.0)),
                progress=False,
            )
        elif method == "teasar":
            skel_obj = sk.skeletonize.by_teasar(
                mesh,
                inv_dist=float(cfg.get("inv_dist", 1000.0)),
                progress=False,
            )
        else:
            skel_obj = sk.skeletonize.by_wavefront(
                mesh,
                waves=int(cfg.get("waves", 2)),
                step_size=int(cfg.get("step_size", 1)),
                progress=False,
            )

        nrn = _skeletor_to_tree_neuron(skel_obj, rid)

        _report(
            progress_callback,
            f"[{idx + 1}/{len(root_ids)}] Healing, pruning twigs, smoothing & resampling {rid}...",
            base_pct + 0.85 * span_pct,
        )
        if eff_heal_dist > 0 and len(nrn.nodes) >= 2:
            try:
                healed = navis.heal_skeleton(nrn, method="ALL", max_dist=eff_heal_dist, inplace=False)
                if healed is not None:
                    nrn = _make_clean_tree_neuron(healed.nodes, rid=rid, name=f"cave_{rid}")
            except Exception:
                pass

        if eff_min_twig > 0 and len(nrn.nodes) >= 4:
            nrn = _prune_spurious_twigs(
                nrn,
                min_twig_nm=eff_min_twig,
                radius_factor=radius_factor,
                max_passes=25,
            )

        if smooth_window >= 3 and len(nrn.nodes) >= 6:
            try:
                smoothed = navis.smooth_skeleton(nrn, window=smooth_window, inplace=False)
                if smoothed is not None:
                    nrn = _make_clean_tree_neuron(smoothed.nodes, rid=rid, name=f"cave_{rid}")
            except Exception:
                pass

        if eff_resample > 0 and len(nrn.nodes) >= 3:
            try:
                resampled = navis.resample_skeleton(nrn, resample_to=eff_resample, inplace=False)
                if resampled is not None and len(resampled.nodes) >= 2:
                    nrn = _make_clean_tree_neuron(resampled.nodes, rid=rid, name=f"cave_{rid}")
            except Exception:
                pass

        # Final pruning pass after smoothing/resampling shrinks curved twigs below min_twig_nm
        if eff_min_twig > 0 and len(nrn.nodes) >= 4:
            nrn = _prune_spurious_twigs(
                nrn,
                min_twig_nm=eff_min_twig,
                radius_factor=radius_factor,
                max_passes=25,
            )

        nodes = nrn.nodes.copy(deep=True)
        nodes[["x", "y", "z"]] = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) / cm_scale
        out[rid] = _make_clean_tree_neuron(nodes, rid=rid, name=f"cave_{rid}")

    return out