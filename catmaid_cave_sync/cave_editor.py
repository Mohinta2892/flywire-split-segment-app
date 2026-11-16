# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:45:38 2026

@author: Pedro Gomez Galvez
"""

from pathlib import Path
import pandas as pd
from caveclient import CAVEclient
from .planner import select_conflict_seeds, find_merge_crossings


def execute_approved_edits(
    mapped_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    merges_df: pd.DataFrame,
    cave_client: CAVEclient,
    cave_resolution_nm: tuple[float, float, float],
    audit_log_path: Path = Path("proofreading_audit_log.csv"),
    progress_callback=None,
) -> pd.DataFrame:
    """Execute approved splits first (using conflict-neck manual seeds), then approved edge merges."""
    df = mapped_df.copy()
    records = []

    approved_splits = splits_df[splits_df["approve"]].copy() if not splits_df.empty else pd.DataFrame()
    approved_merges = merges_df[merges_df["approve"]].copy() if not merges_df.empty else pd.DataFrame()
    total_steps = len(approved_splits) + len(approved_merges)
    step = 0

    def _record(entry: dict):
        records.append(entry)
        row_df = pd.DataFrame([entry])
        row_df.to_csv(
            audit_log_path,
            mode="a",
            header=not audit_log_path.exists(),
            index=False,
        )

    # --- PHASE 1: SPLITS ---
    for _, row in approved_splits.iterrows():
        step += 1
        if progress_callback:
            progress_callback(
                step,
                total_steps,
                f"Splitting root {row['target_root_id']} (skid {row['source_skid']})",
            )

        src_skid = int(row["source_skid"])
        snk_skids = [int(s) for s in str(row["sink_skids"]).split(",") if s.strip()]
        relevant = df[df["skeleton_id"].isin([src_skid] + snk_skids)].copy()

        # Re-resolve live root IDs in case an earlier split updated this root
        relevant["root_id"] = cave_client.chunkedgraph.get_roots(relevant["supervoxel_id"].tolist())
        src_all = relevant[relevant["skeleton_id"] == src_skid]
        snk_all = relevant[relevant["skeleton_id"].isin(snk_skids)]

        shared_roots = set(src_all["root_id"]).intersection(set(snk_all["root_id"]))
        for shared_root in shared_roots:
            src_raw = src_all[src_all["root_id"] == shared_root]
            snk_raw = snk_all[snk_all["root_id"] == shared_root]
            src, snk, _ = select_conflict_seeds(src_raw, snk_raw)

            if src.empty or snk.empty:
                _record({
                    "action": "SPLIT",
                    "status": "SKIPPED",
                    "operation_id": "",
                    "skeleton_id": src_skid,
                    "old_root_id": int(shared_root),
                    "new_root_ids": "",
                    "error": "All overlapping nodes shared the same supervoxel.",
                })
                continue

            try:
                op_id, new_roots = cave_client.chunkedgraph.execute_split(
                    source_points=src[["vx_x", "vx_y", "vx_z"]].to_numpy(),
                    sink_points=snk[["vx_x", "vx_y", "vx_z"]].to_numpy(),
                    root_id=int(shared_root),
                    source_supervoxels=src["supervoxel_id"].tolist(),
                    sink_supervoxels=snk["supervoxel_id"].tolist(),
                )
                _record({
                    "action": "SPLIT",
                    "status": "SUCCESS",
                    "operation_id": op_id,
                    "skeleton_id": src_skid,
                    "old_root_id": int(shared_root),
                    "new_root_ids": ",".join(map(str, new_roots)),
                    "error": "",
                })
            except Exception as exc:
                _record({
                    "action": "SPLIT",
                    "status": "FAILED",
                    "operation_id": "",
                    "skeleton_id": src_skid,
                    "old_root_id": int(shared_root),
                    "new_root_ids": "",
                    "error": str(exc),
                })

    # --- PHASE 2: MERGES (runs only after all splits complete) ---
    if not approved_merges.empty:
        df["root_id"] = cave_client.chunkedgraph.get_roots(df["supervoxel_id"].tolist())

    for _, row in approved_merges.iterrows():
        step += 1
        skid = int(row["skeleton_id"])
        if progress_callback:
            progress_callback(step, total_steps, f"Merging fragments for skeleton {skid}")

        group = df[df["skeleton_id"] == skid].copy()
        group["root_id"] = cave_client.chunkedgraph.get_roots(group["supervoxel_id"].tolist())
        crossings = find_merge_crossings(group)

        for hit in crossings:
            node_a, node_b = hit["node_a"], hit["node_b"]
            sv_pair = [int(node_a["supervoxel_id"]), int(node_b["supervoxel_id"])]
            coord_pair = [
                [int(node_a["vx_x"]), int(node_a["vx_y"]), int(node_a["vx_z"])],
                [int(node_b["vx_x"]), int(node_b["vx_y"]), int(node_b["vx_z"])],
            ]

            try:
                cave_client.chunkedgraph.do_merge(
                    supervoxels=sv_pair,
                    coords=coord_pair,
                    resolution=tuple(cave_resolution_nm),
                )
                merged_root = int(cave_client.chunkedgraph.get_roots([sv_pair[0]])[0])
                _record({
                    "action": "MERGE",
                    "status": "SUCCESS",
                    "operation_id": "",
                    "skeleton_id": skid,
                    "old_root_id": int(hit["fragment_root_id"]),
                    "new_root_ids": str(merged_root),
                    "error": "",
                })
            except Exception as exc:
                _record({
                    "action": "MERGE",
                    "status": "FAILED",
                    "operation_id": "",
                    "skeleton_id": skid,
                    "old_root_id": int(hit["fragment_root_id"]),
                    "new_root_ids": "",
                    "error": str(exc),
                })

    return pd.DataFrame(records)