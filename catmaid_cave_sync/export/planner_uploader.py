"""Build CAVE -> CATMAID export dry-run plans and execute uploads/stitching in CATMAID."""
import inspect

import navis
import numpy as np
import pandas as pd
import pymaid
from caveclient import CAVEclient
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from ..spatial_mapper import map_nodes_to_cave
from .catmaid_matcher import (
    _get_neurons_fast,
    _no_pymaid_cache,
    build_neuron_specific_boxes,
    decimate_subtree_by_slices,
    load_existing_catmaid_trees,
    point_to_segments_min_dist,
    split_kept_into_subtrees,
)
from .retry import retry_call
from .skeletonizer import _make_clean_tree_neuron, _report, skeletonize_cave_roots_skeletor


def _filter_nodes_near_cave_skeletons(
    internal_df: pd.DataFrame,
    cave_skeletons: dict[int, navis.TreeNeuron],
    scale: np.ndarray,
    max_dist_nm: float,
    max_nodes_per_skid: int = 80,
) -> pd.DataFrame:
    """Keep only CATMAID nodes within max_dist_nm of the CAVE skeleton centerlines (sorted closest-first)."""
    if internal_df.empty or not cave_skeletons:
        return internal_df

    cave_chunks = []
    for nrn in cave_skeletons.values():
        nodes = nrn.nodes
        pts = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
        cave_chunks.append(pts)
        by_id = nodes.set_index("node_id")[["x", "y", "z"]]
        valid = nodes[nodes["parent_id"].isin(by_id.index)]
        if not valid.empty:
            a = by_id.loc[valid["parent_id"].to_numpy()].to_numpy(dtype=np.float64) * scale
            b = valid[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale
            dists = np.linalg.norm(b - a, axis=1)
            for idx_e in np.where(dists > 250.0)[0]:
                n_steps = int(np.ceil(dists[idx_e] / 250.0))
                ts = np.linspace(0.0, 1.0, n_steps + 1)[1:-1, None]
                cave_chunks.append(a[idx_e] + ts * (b[idx_e] - a[idx_e]))

    tree = cKDTree(np.vstack(cave_chunks))
    if {"x_nm", "y_nm", "z_nm"}.issubset(internal_df.columns):
        cm_pts_nm = internal_df[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    else:
        cm_pts_nm = internal_df[["x", "y", "z"]].to_numpy(dtype=np.float64) * scale

    nn_dists, _ = tree.query(cm_pts_nm, k=1, distance_upper_bound=float(max_dist_nm))
    keep = np.isfinite(nn_dists)
    if not np.any(keep):
        return internal_df.iloc[0:0].copy()

    filtered = internal_df.loc[keep].copy()
    filtered["_cave_dist_nm"] = nn_dists[keep]
    filtered = filtered.sort_values("_cave_dist_nm", ascending=True)
    if max_nodes_per_skid and max_nodes_per_skid > 0:
        filtered = filtered.groupby("skeleton_id", sort=False).head(int(max_nodes_per_skid))
    return filtered.drop(columns=["_cave_dist_nm"]).reset_index(drop=True)


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
    slice_step: int = 1,
    skeletor_method: str = "wavefront",
    min_twig_nm: float = 500.0,
    resample_to_nm: float = 160.0,
    min_fragment_nodes: int = 3,
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """Skeletonize CAVE roots via skeletor, pair with existing CATMAID nodes, cut overlapping branches & decimate."""
    raw_unique = list(dict.fromkeys(int(r) for r in cave_root_ids))
    scale = (
        np.asarray(catmaid_resolution_nm, dtype=np.float64)
        if catmaid_coord_unit == "voxels"
        else np.ones(3, dtype=np.float64)
    )

    _report(progress_callback, "Verifying latest CAVE root IDs...", 4.0)

    is_latest_flags = retry_call(
        lambda: cave_client.chunkedgraph.is_latest_roots(raw_unique),
        stage="CAVE is_latest_roots",
        progress_callback=progress_callback,
    )
    latest_roots: list[int] = []
    for rid, is_lat in zip(raw_unique, is_latest_flags):
        if not is_lat:
            latest = retry_call(
                lambda r=rid: cave_client.chunkedgraph.get_latest_roots(r),
                stage=f"CAVE get_latest_roots({rid})",
                progress_callback=progress_callback,
            )
            rid = int(latest[0])
        if rid not in latest_roots:
            latest_roots.append(rid)

    cave_skeletons = skeletonize_cave_roots_skeletor(
        cave_client=cave_client,
        root_ids=latest_roots,
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
        skeletor_method=skeletor_method,
        min_twig_nm=min_twig_nm,
        resample_to_nm=resample_to_nm,
        progress_callback=progress_callback,
        pct_range=(8.0, 62.0),
    )

    local_boxes_nm = None
    if not search_annotations.strip():
        local_boxes_nm = build_neuron_specific_boxes(
            cave_skeletons,
            scale=scale,
            pad_nm=float(cut_buffer_nm) + 150.0,
        )
        _report(
            progress_callback,
            f"Querying CATMAID neighborhood ({len(local_boxes_nm)} tight local sub-boxes)...",
            66.0,
        )
    else:
        _report(progress_callback, "Loading existing CATMAID skeletons from search annotations...", 66.0)

    trees_by_skid, internal_df = load_existing_catmaid_trees(
        rm,
        search_annotations=search_annotations,
        local_boxes_nm=local_boxes_nm,
        catmaid_resolution_nm=catmaid_resolution_nm,
        catmaid_coord_unit=catmaid_coord_unit,
        progress_callback=progress_callback,
    )

    min_match_nodes = max(2, int(min_fragment_nodes))
    root_to_skids: dict[int, list[tuple[int, int]]] = {}
    if not internal_df.empty:
        near_df = _filter_nodes_near_cave_skeletons(
            internal_df,
            cave_skeletons=cave_skeletons,
            scale=scale,
            max_dist_nm=max(float(cut_buffer_nm) + 600.0, 1200.0),
            max_nodes_per_skid=80,
        )
        if not near_df.empty:
            _report(
                progress_callback,
                f"Mapping {len(near_df)} near-neuron CATMAID nodes (filtered from {len(internal_df)}) to CAVE roots...",
                80.0,
            )
            mapped_cm = retry_call(
                lambda: map_nodes_to_cave(
                    near_df,
                    cave_client,
                    catmaid_resolution_nm=catmaid_resolution_nm,
                    cave_resolution_nm=cave_resolution_nm,
                    catmaid_coord_unit=catmaid_coord_unit,
                ),
                stage="CAVE map_nodes_to_cave",
                progress_callback=progress_callback,
            )
            for (r_id, skid), grp in mapped_cm.groupby(["root_id", "skeleton_id"]):
                n_overlap = int(len(grp))
                if n_overlap >= min_match_nodes:
                    root_to_skids.setdefault(int(r_id), []).append((int(skid), n_overlap))

    _report(progress_callback, "Subtracting overlapping branches & formatting subtrees...", 92.0)

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
            existing_chunks = []
            for s in matched_skids:
                df_s = trees_by_skid[s]["existing_nodes"].copy()
                df_s["skeleton_id"] = int(s)
                existing_chunks.append(df_s)
            combined_existing = pd.concat(existing_chunks, ignore_index=True)
            min_dists = point_to_segments_min_dist(
                cave_pts_nm,
                combined_existing,
                max_query_radius_nm=float(cut_buffer_nm) + 50.0,
            )
            keep_mask = min_dists > float(cut_buffer_nm)

        if user_base:
            base_name = user_base.replace("{n}", str(root_idx)) if "{n}" in user_base else user_base
        elif master_skid is not None:
            base_name = trees_by_skid[master_skid]["name"]
        else:
            base_name = f"cave_{rid}"

        raw_subtrees = split_kept_into_subtrees(nrn, keep_mask, min_fragment_nodes=min_fragment_nodes)
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
                nearest_skid = int(combined_existing.iloc[man_j]["skeleton_id"])
                anchor_info = {
                    "sub_local_node_id": int(sub_df.iloc[sub_i]["node_id"]),
                    "master_skid": nearest_skid,
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
    pair = retry_call(
        lambda: _get_neurons_fast(rm, [int(master_skid), int(child_skid)]),
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

    with _no_pymaid_cache(rm):
        url = rm.make_url(rm.project_id, "skeleton", "join")
        try:
            retry_call(
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
        except Exception as rest_exc:
            if not hasattr(pymaid, "join_skeletons"):
                raise
            sig = inspect.signature(pymaid.join_skeletons)
            kw = {"remote_instance": rm}
            if "winner" in sig.parameters:
                kw["winner"] = int(master_skid)
            retry_call(
                lambda: pymaid.join_skeletons([int(master_skid), int(child_skid)], **kw),
                stage=f"CATMAID join_skeletons (after {rest_exc})",
            )
    return min_dist


def execute_cave_export_plan(
    plan_rows: list[dict],
    export_bundle: dict,
    rm: pymaid.CatmaidInstance,
    target_annotations: str = "",
    auto_stitch: bool = False,
    max_stitch_dist_nm: float = 2000.0,
    progress_callback=None,
) -> list[dict]:
    """Upload approved non-overlapping CAVE fragments to CATMAID without node tags, optionally stitch & annotate."""
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
    n_app = max(1, len(approved))

    with _no_pymaid_cache(rm):
        for idx, row in enumerate(approved, start=1):
            rid_str = str(row["cave_root_id"])
            item = items.get(rid_str)
            if not item or not item["fragments"]:
                continue

            base_pct = 5.0 + ((idx - 1) / n_app) * 90.0
            span_pct = 90.0 / n_app
            _report(progress_callback, f"[{idx}/{len(approved)}] Uploading CAVE root {rid_str} to CATMAID...", base_pct)

            edited_names = [x.strip() for x in str(row.get("fragment_names") or "").split(",") if x.strip()]
            stitched_skids = []
            created_skids = []
            created_names = []
            nodes_added = 0
            notes = []

            try:
                n_frags = max(1, len(item["fragments"]))
                for f_idx, frag in enumerate(item["fragments"]):
                    frag_name = (
                        edited_names[f_idx]
                        if f_idx < len(edited_names)
                        else (edited_names[0] if len(edited_names) == 1 and len(item["fragments"]) == 1 else frag["name"])
                    )
                    sub_df = frag["nodes_df"]
                    nrn = _make_clean_tree_neuron(sub_df, rid=0, name=frag_name)

                    _report(
                        progress_callback,
                        f"[{idx}/{len(approved)}] Uploading {frag_name} ({len(sub_df)} nodes)...",
                        base_pct + (f_idx / n_frags) * 0.7 * span_pct,
                    )
                    up_res = retry_call(
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
                    target_skid = (
                        int(anchor["master_skid"])
                        if anchor and anchor.get("master_skid") is not None
                        else item["master_skid"]
                    )
                    anchor_dist = (
                        float(anchor["anchor_dist_nm"])
                        if anchor and anchor.get("anchor_dist_nm") is not None
                        else None
                    )

                    if auto_stitch and target_skid is not None and anchor_dist is not None:
                        if anchor_dist > float(max_stitch_dist_nm):
                            created_skids.append(new_skid)
                            created_names.append(frag_name)
                            notes.append(
                                f"Kept {frag_name} (SKID {new_skid}) separate ({anchor_dist:.0f} nm from SKID {target_skid})"
                            )
                        else:
                            _report(
                                progress_callback,
                                f"[{idx}/{len(approved)}] Stitching {frag_name} into SKID {target_skid}...",
                                base_pct + ((f_idx + 0.5) / n_frags) * 0.7 * span_pct,
                            )
                            try:
                                dist_nm = _join_catmaid_skeletons(
                                    rm,
                                    master_skid=int(target_skid),
                                    child_skid=int(new_skid),
                                    scale=scale,
                                    max_stitch_dist_nm=max_stitch_dist_nm,
                                )
                                stitched_skids.append(int(target_skid))
                                notes.append(f"Stitched {frag_name} into SKID {target_skid} ({dist_nm:.0f} nm)")
                            except Exception as join_exc:
                                created_skids.append(new_skid)
                                created_names.append(frag_name)
                                notes.append(f"Kept {frag_name} (SKID {new_skid}) separate: {join_exc}")
                    else:
                        created_skids.append(new_skid)
                        created_names.append(frag_name)

                final_skids = list(dict.fromkeys([*stitched_skids, *created_skids]))
                if not final_skids and item["master_skid"] is not None:
                    final_skids = [int(item["master_skid"])]

                per_neuron_anns = list(dict.fromkeys([*ann_list, rid_str])) if rid_str else list(ann_list)
                if per_neuron_anns and final_skids:
                    _report(
                        progress_callback,
                        f"[{idx}/{len(approved)}] Applying CATMAID annotations...",
                        base_pct + 0.85 * span_pct,
                    )
                    retry_call(
                        lambda anns=per_neuron_anns: pymaid.add_annotations(final_skids, anns, remote_instance=rm),
                        stage="CATMAID add_annotations",
                        progress_callback=progress_callback,
                    )

                audit.append({
                    "cave_root_id": rid_str,
                    "status": "SUCCESS",
                    "resulting_skids": ",".join(map(str, final_skids)),
                    "fragment_names": ", ".join(created_names) if created_names else f"Merged into {item['base_name']}",
                    "nodes_added": nodes_added,
                    "annotations": ", ".join(per_neuron_anns),
                    "details": "; ".join(notes) if notes else "Uploaded cleanly",
                })
            except Exception as exc:
                audit.append({
                    "cave_root_id": rid_str,
                    "status": "FAILED",
                    "resulting_skids": ",".join(map(str, [*stitched_skids, *created_skids])),
                    "fragment_names": ", ".join(created_names),
                    "nodes_added": nodes_added,
                    "annotations": "",
                    "details": str(exc),
                })

    return audit