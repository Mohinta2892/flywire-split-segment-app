# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:32:15 2026

@author: Pedro Gomez Galvez
"""

from pathlib import Path
import pandas as pd
import pymaid
import navis


def fetch_and_cache_skeletons(
    skids_or_annotation: list[int] | str,
    cache_dir: Path = Path("data/catmaid_skeletons"),
    force_refresh: bool = False,
) -> navis.NeuronList:
    """Download CATMAID skeletons (by ID list or annotation) and cache locally as SWC."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(skids_or_annotation, str):
        skids = [int(s) for s in pymaid.get_skids_by_annotation(skids_or_annotation)]
    else:
        skids = [int(s) for s in skids_or_annotation]

    neurons = []
    missing_skids = []

    for skid in skids:
        swc_path = cache_dir / f"{skid}.swc"
        if swc_path.exists() and not force_refresh:
            n = navis.read_swc(swc_path)
            n.id = skid
            neurons.append(n)
        else:
            missing_skids.append(skid)

    if missing_skids:
        fetched = pymaid.get_neurons(missing_skids)
        for n in fetched:
            n.to_swc(cache_dir / f"{int(n.id)}.swc")
            neurons.append(n)

    return navis.NeuronList(neurons)


def extract_non_end_nodes(neurons: navis.NeuronList) -> pd.DataFrame:
    """Keep all manually placed nodes EXCEPT end/leaf nodes, preserving parent_id edges."""
    records = []
    for n in neurons:
        nodes = n.nodes.copy()
        parent_ids = set(nodes["parent_id"].dropna().astype(int))
        kept = nodes[nodes["node_id"].isin(parent_ids)].copy()

        # Fallback only if a skeleton is a 1-node stub so it isn't silently lost
        if kept.empty:
            kept = nodes.copy()

        kept["skeleton_id"] = int(n.id)
        kept["node_id"] = kept["node_id"].astype(int)
        kept["parent_id"] = kept["parent_id"].fillna(-1).astype(int)
        records.append(kept[["skeleton_id", "node_id", "parent_id", "x", "y", "z"]])

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()