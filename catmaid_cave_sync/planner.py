# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:53:08 2026

@author: Pedro Gomez Galvez
"""

import json
import urllib.parse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from caveclient import CAVEclient


def _build_segments(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build (start_nm, end_nm) arrays for all internal parent->child edges in df."""
    if len(df) < 2 or "parent_id" not in df.columns:
        return np.empty((0, 3)), np.empty((0, 3))
    by_id = df.set_index("node_id")[["x_nm", "y_nm", "z_nm"]]
    valid = df[df["parent_id"].isin(by_id.index)]
    if valid.empty:
        return np.empty((0, 3)), np.empty((0, 3))
    p_coords = by_id.loc[valid["parent_id"].to_numpy()].to_numpy(dtype=np.float64)
    c_coords = valid[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    return p_coords, c_coords


def _point_to_trajectory_distances(pts_nm: np.ndarray, target_df: pd.DataFrame) -> np.ndarray:
    """Minimum 3D distance (nm) from each manual point in pts_nm to target_df's traced segments."""
    seg_a, seg_b = _build_segments(target_df)
    if len(seg_a) == 0:
        target_pts = target_df[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
        return cdist(pts_nm, target_pts).min(axis=1)

    ab = seg_b - seg_a                                  # (M, 3)
    ab_len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)  # (M,)
    ap = pts_nm[:, None, :] - seg_a[None, :, :]         # (N, M, 3)
    t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
    closest = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]
    dists = np.linalg.norm(pts_nm[:, None, :] - closest, axis=2)
    return dists.min(axis=1)


def select_conflict_seeds(
    src_df: pd.DataFrame,
    snk_df: pd.DataFrame,
    local_seeds: int = 35,
    anchor_seeds: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Scrub shared supervoxels and select manual nodes concentrated around the conflict neck."""
    shared_svs = set(src_df["supervoxel_id"]).intersection(set(snk_df["supervoxel_id"]))
    src_clean = src_df[~src_df["supervoxel_id"].isin(shared_svs)].copy()
    snk_clean = snk_df[~snk_df["supervoxel_id"].isin(shared_svs)].copy()

    if src_clean.empty or snk_clean.empty:
        return src_clean, snk_clean, float("inf")

    src_pts = src_clean[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    snk_pts = snk_clean[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)

    src_dists = _point_to_trajectory_distances(src_pts, snk_clean)
    snk_dists = _point_to_trajectory_distances(snk_pts, src_clean)
    min_conflict_nm = float(min(src_dists.min(), snk_dists.min()))

    def _pick_subset(df: pd.DataFrame, dists: np.ndarray) -> pd.DataFrame:
        order = np.argsort(dists)
        if len(order) <= (local_seeds + anchor_seeds):
            return df.iloc[order]
        near_idx = order[:local_seeds]
        far_pool = order[local_seeds:]
        step_idx = far_pool[np.linspace(0, len(far_pool) - 1, anchor_seeds, dtype=int)]
        chosen = np.unique(np.concatenate([near_idx, step_idx]))
        return df.iloc[chosen]

    return _pick_subset(src_clean, src_dists), _pick_subset(snk_clean, snk_dists), min_conflict_nm


def find_merge_crossings(group: pd.DataFrame) -> list[dict]:
    """Find false-split crossings for one skeleton using CATMAID parent->child edges first."""
    root_counts = group["root_id"].value_counts()
    if len(root_counts) <= 1:
        return []

    primary_root = int(root_counts.index[0])
    primary_n = int(root_counts[primary_root])
    by_id = group.set_index("node_id")

    # Map every internal parent->child edge whose endpoints land in different CAVE roots
    edge_crossings = {}
    if "parent_id" in group.columns:
        for _, child in group.iterrows():
            pid = int(child["parent_id"])
            if pid not in by_id.index:
                continue
            parent = by_id.loc[pid]
            r_p, r_c = int(parent["root_id"]), int(child["root_id"])
            if r_p == r_c:
                continue
            gap_nm = float(np.linalg.norm([
                parent["x_nm"] - child["x_nm"],
                parent["y_nm"] - child["y_nm"],
                parent["z_nm"] - child["z_nm"],
            ]))
            pair_key = tuple(sorted((r_p, r_c)))
            if pair_key not in edge_crossings or gap_nm < edge_crossings[pair_key]["gap_nm"]:
                edge_crossings[pair_key] = {
                    "node_a": parent,
                    "node_b": child,
                    "gap_nm": gap_nm,
                    "source": "traced_edge",
                }

    crossings = []
    for frag_root in [int(r) for r in root_counts.index[1:]]:
        frag_n = int(root_counts[frag_root])
        pair_key = tuple(sorted((primary_root, frag_root)))
        if pair_key in edge_crossings:
            hit = edge_crossings[pair_key]
            crossings.append({
                "primary_root_id": primary_root,
                "fragment_root_id": frag_root,
                "primary_nodes": primary_n,
                "fragment_nodes": frag_n,
                "gap_nm": round(hit["gap_nm"], 1),
                "bridge_type": "traced_edge",
                "node_a": hit["node_a"],
                "node_b": hit["node_b"],
            })
        else:
            # Fallback if the skeleton has multiple disconnected components in CATMAID
            p_nodes = group[group["root_id"] == primary_root]
            f_nodes = group[group["root_id"] == frag_root]
            dists = cdist(
                p_nodes[["x_nm", "y_nm", "z_nm"]].to_numpy(),
                f_nodes[["x_nm", "y_nm", "z_nm"]].to_numpy(),
            )
            i_idx, j_idx = np.unravel_index(np.argmin(dists), dists.shape)
            crossings.append({
                "primary_root_id": primary_root,
                "fragment_root_id": frag_root,
                "primary_nodes": primary_n,
                "fragment_nodes": frag_n,
                "gap_nm": round(float(dists[i_idx, j_idx]), 1),
                "bridge_type": "nearest_nodes",
                "node_a": p_nodes.iloc[i_idx],
                "node_b": f_nodes.iloc[j_idx],
            })
    return crossings


def build_proofreading_plan(
    mapped_df: pd.DataFrame,
    min_support_nodes: int = 3,
    max_merge_gap_nm: float = 1500.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Generate dry-run SPLIT and MERGE plans using trajectory conflict detection and traced edges."""
    if mapped_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    # 1. Planned SPLITS: CAVE roots overlapping >1 CATMAID skeleton
    split_rows = []
    for root_id, group in mapped_df.groupby("root_id"):
        skid_counts = group["skeleton_id"].value_counts()
        if len(skid_counts) <= 1:
            continue

        ordered_skids = skid_counts.index.tolist()
        for idx in range(len(ordered_skids) - 1):
            src_skid = int(ordered_skids[idx])
            snk_skids = [int(s) for s in ordered_skids[idx + 1 :]]
            src_df = group[group["skeleton_id"] == src_skid]
            snk_df = group[group["skeleton_id"].isin(snk_skids)]

            src_seeds, snk_seeds, conflict_nm = select_conflict_seeds(src_df, snk_df)
            src_n = len(src_seeds)
            snk_n = len(snk_seeds)

            split_rows.append({
                "approve": (src_n >= min_support_nodes and snk_n >= min_support_nodes),
                "action": "SPLIT",
                "target_root_id": int(root_id),
                "source_skid": src_skid,
                "sink_skids": ",".join(map(str, snk_skids)),
                "source_nodes": src_n,
                "sink_nodes": snk_n,
                "conflict_dist_nm": round(conflict_nm, 1) if np.isfinite(conflict_nm) else -1.0,
            })

    splits_df = pd.DataFrame(split_rows)

    # 2. Planned MERGES: CATMAID skeletons spanning >1 CAVE root
    merge_rows = []
    for skid, group in mapped_df.groupby("skeleton_id"):
        for hit in find_merge_crossings(group):
            enough_nodes = (
                hit["primary_nodes"] >= min_support_nodes
                and hit["fragment_nodes"] >= min_support_nodes
            )
            within_gap = hit["gap_nm"] <= float(max_merge_gap_nm)
            merge_rows.append({
                "approve": bool(enough_nodes and within_gap),
                "action": "MERGE",
                "skeleton_id": int(skid),
                "primary_root_id": hit["primary_root_id"],
                "fragment_root_id": hit["fragment_root_id"],
                "primary_nodes": hit["primary_nodes"],
                "fragment_nodes": hit["fragment_nodes"],
                "gap_nm": hit["gap_nm"],
                "bridge_type": hit["bridge_type"],
            })

    merges_df = pd.DataFrame(merge_rows)

    # 3. Batch Summary Metrics
    skid_root_counts = mapped_df.groupby("skeleton_id")["root_id"].nunique()
    root_skid_counts = mapped_df.groupby("root_id")["skeleton_id"].nunique()
    clean_skids = [
        s for s, n_roots in skid_root_counts.items()
        if n_roots == 1
        and root_skid_counts[mapped_df[mapped_df["skeleton_id"] == s]["root_id"].iloc[0]] == 1
    ]

    summary = {
        "total_skeletons": int(mapped_df["skeleton_id"].nunique()),
        "total_mapped_nodes": int(len(mapped_df)),
        "clean_1to1_skeletons": len(clean_skids),
        "planned_splits": len(splits_df),
        "planned_merges": len(merges_df),
    }
    return summary, splits_df, merges_df


def preview_single_split(
    mapped_df: pd.DataFrame,
    cave_client: CAVEclient,
    root_id: int,
    source_skid: int,
    sink_skids: list[int],
    cave_resolution_nm: tuple[float, float, float],
) -> dict:
    """Run CAVE preview_split using scrubbed conflict-zone seeds and build a 3D Neuroglancer URL."""
    sub = mapped_df[mapped_df["root_id"] == int(root_id)]
    src_raw = sub[sub["skeleton_id"] == int(source_skid)]
    snk_raw = sub[sub["skeleton_id"].isin([int(s) for s in sink_skids])]

    src, snk, _ = select_conflict_seeds(src_raw, snk_raw)
    if src.empty or snk.empty:
        raise ValueError("No distinct non-overlapping supervoxels remain between source and sink.")

    src_pts = src[["vx_x", "vx_y", "vx_z"]].to_numpy()
    snk_pts = snk[["vx_x", "vx_y", "vx_z"]].to_numpy()

    new_components = cave_client.chunkedgraph.preview_split(
        source_points=src_pts,
        sink_points=snk_pts,
        root_id=int(root_id),
        source_supervoxels=src["supervoxel_id"].tolist(),
        sink_supervoxels=snk["supervoxel_id"].tolist(),
    )

    state = {
        "dimensions": {
            "x": [cave_resolution_nm[0] * 1e-9, "m"],
            "y": [cave_resolution_nm[1] * 1e-9, "m"],
            "z": [cave_resolution_nm[2] * 1e-9, "m"],
        },
        "position": src_pts[0].tolist(),
        "layers": [
            {
                "type": "segmentation",
                "source": cave_client.info.segmentation_source(),
                "name": "segmentation",
                "segments": [str(root_id)],
            },
            {
                "type": "annotation",
                "name": f"source_skid_{source_skid}",
                "annotationColor": "#00ff00",
                "annotations": [
                    {"point": p.tolist(), "type": "point", "id": f"src_{i}"}
                    for i, p in enumerate(src_pts)
                ],
            },
            {
                "type": "annotation",
                "name": f"sink_skids_{'_'.join(map(str, sink_skids))}",
                "annotationColor": "#ff0000",
                "annotations": [
                    {"point": p.tolist(), "type": "point", "id": f"snk_{i}"}
                    for i, p in enumerate(snk_pts)
                ],
            },
        ],
    }
    ngl_base = cave_client.info.viewer_site() or "https://neuroglancer-demo.appspot.com"
    ngl_url = f"{ngl_base}/#!{urllib.parse.quote(json.dumps(state))}"

    return {
        "split_valid": len(new_components) >= 2,
        "predicted_component_count": len(new_components),
        "neuroglancer_url": ngl_url,
    }