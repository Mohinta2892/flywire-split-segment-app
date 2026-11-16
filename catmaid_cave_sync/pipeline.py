# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:46:22 2026

@author: Pedro Gomez Galvez
"""

import pandas as pd
from .config import load_or_create_config, init_clients
from .catmaid_loader import fetch_and_cache_skeletons, extract_internal_nodes
from .spatial_mapper import map_nodes_to_cave
from .cave_editor import split_false_merges, merge_false_splits


def run_catmaid_driven_proofreading(
    skids_or_annotation: list[int] | str,
    dry_run: bool = True,
    force_refresh_skeletons: bool = False,
) -> pd.DataFrame:
    config = load_or_create_config()
    cave_client, _ = init_clients(config)
    resolution = tuple(config["voxel_resolution"])

    neurons = fetch_and_cache_skeletons(
        skids_or_annotation,
        force_refresh=force_refresh_skeletons,
    )
    internal_nodes = extract_internal_nodes(neurons)
    mapped_df = map_nodes_to_cave(internal_nodes, cave_client, resolution=resolution)

    split_logs = split_false_merges(mapped_df, cave_client, resolution=resolution, dry_run=dry_run)
    merge_logs = merge_false_splits(mapped_df, cave_client, resolution=resolution, dry_run=dry_run)

    audit_df = pd.DataFrame(split_logs + merge_logs)
    audit_df.to_csv("proofreading_audit_log.csv", index=False)
    return audit_df