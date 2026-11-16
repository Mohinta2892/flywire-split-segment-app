# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:37:24 2026

@author: Pedro Gomez Galvez
"""

from pathlib import Path
import streamlit as st
from caveclient import CAVEclient
import pymaid

from catmaid_cave_sync.config import (
    load_all_configs,
    save_project_config,
    query_cave_resolution,
    query_catmaid_stacks,
    init_clients_from_project,
)
from catmaid_cave_sync.catmaid_loader import (
    fetch_and_cache_skeletons,
    extract_non_end_nodes,
)
from catmaid_cave_sync.spatial_mapper import map_nodes_to_cave
from catmaid_cave_sync.planner import build_proofreading_plan, preview_single_split
from catmaid_cave_sync.cave_editor import execute_approved_edits


def _parse_resolution_csv(raw_str: str, label: str) -> list[float]:
    """Parse a 'x,y,z' resolution string into 3 positive floats."""
    parts = [p.strip() for p in raw_str.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{label} must have exactly 3 comma-separated numbers (e.g., 16,16,16).")
    vals = [float(p) for p in parts]
    if any(v <= 0 for v in vals):
        raise ValueError(f"{label} values must all be positive.")
    return vals


def render_project_selector() -> tuple[str | None, dict | None]:
    """Select an existing project profile or query & save a new one on first run."""
    store = load_all_configs()
    saved_projects = list(store["projects"].keys())

    options = saved_projects + ["+ Add New Project Profile"]
    default_idx = (
        saved_projects.index(store["active_project"])
        if store["active_project"] in saved_projects
        else 0
    )
    selected = st.selectbox(
        "Active Volume / Project Profile",
        options=options,
        index=default_idx,
        key="active_project_selector",
    )

    # --- EXISTING SAVED PROJECT ---
    if selected != "+ Add New Project Profile":
        cfg = store["projects"][selected]
        st.caption(
            f"Loaded saved profile **{selected}** | "
            f"Datastack: `{cfg['cave_datastack']}` | "
            f"CATMAID res: `{cfg['catmaid_resolution_nm']}` nm | "
            f"CAVE res: `{cfg['cave_resolution_nm']}` nm"
        )

        with st.expander("Update Tokens or Resolutions for This Project", expanded=False):
            with st.form(f"edit_project_form_{selected}"):
                c1, c2 = st.columns(2)
                with c1:
                    upd_cave_ds = st.text_input("CAVE Datastack", value=cfg["cave_datastack"])
                    upd_cave_tok = st.text_input("CAVE API Token", value=cfg["cave_token"], type="password")
                    upd_cave_res = st.text_input(
                        "CAVE Resolution (x,y,z nm)",
                        value=",".join(map(str, cfg["cave_resolution_nm"])),
                    )
                with c2:
                    upd_cm_url = st.text_input("CATMAID Server URL", value=cfg["catmaid_url"])
                    upd_cm_pid = st.number_input(
                        "CATMAID Project ID",
                        min_value=1,
                        value=int(cfg["catmaid_project_id"]),
                        step=1,
                    )
                    upd_cm_tok = st.text_input("CATMAID API Token", value=cfg["catmaid_token"], type="password")
                    upd_cm_res = st.text_input(
                        "CATMAID Resolution (x,y,z nm)",
                        value=",".join(map(str, cfg["catmaid_resolution_nm"])),
                    )
                    upd_unit = st.selectbox(
                        "CATMAID Coordinate Unit",
                        ["nm", "voxels"],
                        index=0 if cfg.get("catmaid_coord_unit", "nm") == "nm" else 1,
                    )

                if st.form_submit_button("Save Updated Profile"):
                    try:
                        updated_cfg = {
                            "cave_datastack": upd_cave_ds.strip(),
                            "cave_token": upd_cave_tok.strip(),
                            "cave_resolution_nm": _parse_resolution_csv(upd_cave_res, "CAVE resolution"),
                            "catmaid_url": upd_cm_url.strip(),
                            "catmaid_project_id": int(upd_cm_pid),
                            "catmaid_token": upd_cm_tok.strip(),
                            "catmaid_resolution_nm": _parse_resolution_csv(upd_cm_res, "CATMAID resolution"),
                            "catmaid_coord_unit": upd_unit,
                        }
                        save_project_config(selected, updated_cfg)
                        st.success(f"Updated '{selected}' in .cave_catmaid_config.json")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not save updates: {exc}")

        return selected, cfg

    # --- NEW PROJECT FIRST-TIME SETUP ---
    with st.container(border=True):
        st.subheader("First-Time Project Setup")
        proj_name = st.text_input(
            "Project Name (e.g., Parker)",
            value="",
            key="new_proj_name",
        )
        c1, c2 = st.columns(2)
        with c1:
            cave_datastack = st.text_input("CAVE Datastack", key="new_cave_datastack")
            cave_token = st.text_input("CAVE API Token", type="password", key="new_cave_token")
        with c2:
            catmaid_url = st.text_input("CATMAID Server URL", key="new_catmaid_url")
            catmaid_pid = st.number_input("CATMAID Project ID", min_value=1, step=1, key="new_catmaid_pid")
            catmaid_token = st.text_input("CATMAID API Token", type="password", key="new_catmaid_token")

        if st.button("1. Connect & Query Volume Resolutions", key="btn_probe_resolutions"):
            if not proj_name.strip():
                st.error("Please enter a Project Name before connecting.")
                return None, None
            try:
                cave_client = CAVEclient(
                    datastack_name=cave_datastack.strip(),
                    auth_token=cave_token.strip(),
                )
                rm = pymaid.CatmaidInstance(
                    server=catmaid_url.strip(),
                    api_token=catmaid_token.strip(),
                    project_id=int(catmaid_pid),
                )
                queried_cave_res = query_cave_resolution(cave_client)
                queried_stacks = query_catmaid_stacks(rm)
                st.session_state["pending_project_probe"] = {
                    "proj_name": proj_name.strip(),
                    "cave_datastack": cave_datastack.strip(),
                    "cave_token": cave_token.strip(),
                    "catmaid_url": catmaid_url.strip(),
                    "catmaid_project_id": int(catmaid_pid),
                    "catmaid_token": catmaid_token.strip(),
                    "queried_cave_res": queried_cave_res,
                    "queried_stacks": queried_stacks,
                }
            except Exception as exc:
                st.error(f"Connection or resolution query failed: {exc}")

        if "pending_project_probe" in st.session_state:
            probe = st.session_state["pending_project_probe"]
            st.success("Connected! Verify the queried voxel resolutions below and save:")

            cave_res_input = st.text_input(
                "CAVE Segmentation Resolution (x,y,z nm)",
                value=",".join(map(str, probe["queried_cave_res"])),
                key="probe_cave_res_input",
            )

            stacks = probe["queried_stacks"]
            if stacks:
                stack_choice = st.selectbox(
                    "CATMAID Image Stack (auto-queried)",
                    options=range(len(stacks)),
                    format_func=lambda i: (
                        f"{stacks[i]['title']} (ID {stacks[i]['stack_id']}) — "
                        f"{stacks[i]['resolution_nm']} nm"
                    ),
                    key="probe_stack_choice",
                )
                default_catmaid_res = ",".join(map(str, stacks[stack_choice]["resolution_nm"]))
            else:
                st.warning("No stack resolutions returned by CATMAID; enter manually below.")
                default_catmaid_res = ""

            catmaid_res_input = st.text_input(
                "CATMAID Voxel Resolution (x,y,z nm)",
                value=default_catmaid_res,
                key="probe_catmaid_res_input",
            )
            coord_unit = st.selectbox(
                "CATMAID Skeleton Coordinate Unit",
                ["nm", "voxels"],
                index=0,
                key="probe_coord_unit",
            )

            if st.button("2. Save Project Profile Locally", type="primary", key="btn_save_project"):
                final_name = proj_name.strip() or probe["proj_name"]
                if not final_name:
                    st.error("Please enter a Project Name (e.g., Parker).")
                    return None, None
                try:
                    cave_res_vals = _parse_resolution_csv(cave_res_input, "CAVE resolution")
                    catmaid_res_vals = _parse_resolution_csv(catmaid_res_input, "CATMAID resolution")
                except ValueError as exc:
                    st.error(str(exc))
                    return None, None

                save_project_config(
                    final_name,
                    {
                        "cave_datastack": probe["cave_datastack"],
                        "cave_token": probe["cave_token"],
                        "cave_resolution_nm": cave_res_vals,
                        "catmaid_url": probe["catmaid_url"],
                        "catmaid_project_id": probe["catmaid_project_id"],
                        "catmaid_token": probe["catmaid_token"],
                        "catmaid_resolution_nm": catmaid_res_vals,
                        "catmaid_coord_unit": coord_unit,
                    },
                )
                del st.session_state["pending_project_probe"]
                st.success(f"Saved '{final_name}' to .cave_catmaid_config.json")
                st.rerun()

    return None, None


def render_catmaid_proofreading_tab():
    """Main Streamlit UI tab for CATMAID-driven CAVE splitting and merging."""
    st.header("CATMAID-Driven Split & Merge")

    # 1. Project & Volume Resolution Profile
    project_name, config = render_project_selector()
    if not config or not project_name:
        st.info("Select an existing project profile or complete first-time setup above.")
        return

    # Clear cached batch results if the user switched to a different volume/project
    if st.session_state.get("loaded_project") != project_name:
        for key in ("mapped_df", "summary", "splits_df", "merges_df"):
            st.session_state.pop(key, None)
        st.session_state["loaded_project"] = project_name

    # 2. Load Skeletons & Map Overlaps
    st.subheader("Load CATMAID Skeletons & Analyze Overlaps")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query_mode = st.radio(
            "Load skeletons by:",
            ["CATMAID Annotation", "Skeleton ID List"],
            horizontal=True,
            key="skeleton_query_mode",
        )
        query_val = st.text_input(
            "Annotation name or comma-separated Skeleton IDs",
            placeholder="e.g., Parker_proofread_batch1 or 12345, 67890",
            key="skeleton_query_val",
        )
    with c2:
        min_nodes = st.number_input(
            "Auto-approve min nodes/side",
            min_value=1,
            value=3,
            step=1,
            key="min_support_nodes",
        )
    with c3:
        force_refresh = st.checkbox(
            "Re-download SWCs",
            value=False,
            key="force_refresh_swcs",
        )

    if st.button("Download Skeletons & Build Dry-Run Plan", type="primary", key="btn_build_plan"):
        if not query_val.strip():
            st.warning("Enter a CATMAID annotation or at least one Skeleton ID.")
            return

        try:
            cave_client, _ = init_clients_from_project(config)
            if query_mode == "CATMAID Annotation":
                skids_or_ann = query_val.strip()
            else:
                skids_or_ann = [int(x.strip()) for x in query_val.split(",") if x.strip()]

            project_cache_dir = Path("data/catmaid_skeletons") / project_name

            with st.spinner("Downloading/loading local CATMAID skeletons & mapping CAVE roots..."):
                neurons = fetch_and_cache_skeletons(
                    skids_or_ann,
                    cache_dir=project_cache_dir,
                    force_refresh=force_refresh,
                )
                if len(neurons) == 0:
                    st.warning("No CATMAID skeletons matched your query.")
                    return

                internal_nodes = extract_non_end_nodes(neurons)
                mapped_df = map_nodes_to_cave(
                    internal_nodes,
                    cave_client,
                    catmaid_resolution_nm=config["catmaid_resolution_nm"],
                    cave_resolution_nm=config["cave_resolution_nm"],
                    catmaid_coord_unit=config.get("catmaid_coord_unit", "nm"),
                )
                summary, splits_df, merges_df = build_proofreading_plan(
                    mapped_df,
                    min_support_nodes=int(min_nodes),
                )

            st.session_state["mapped_df"] = mapped_df
            st.session_state["summary"] = summary
            st.session_state["splits_df"] = splits_df
            st.session_state["merges_df"] = merges_df
        except Exception as exc:
            st.error(f"Error while loading or mapping skeletons: {exc}")

    # 3. Interactive Batch Preview & Single-Neuron 3D Check
    if "summary" in st.session_state:
        s = st.session_state["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Skeletons Loaded", s.get("total_skeletons", 0))
        m2.metric("Clean 1-to-1", s.get("clean_1to1_skeletons", 0))
        m3.metric("Planned Splits", s.get("planned_splits", 0))
        m4.metric("Planned Merges", s.get("planned_merges", 0))

        splits_df = st.session_state["splits_df"]
        merges_df = st.session_state["merges_df"]

        if not splits_df.empty:
            st.markdown("**Review & Approve Splits (uncheck low-confidence rows to skip):**")
            edited_splits = st.data_editor(
                splits_df,
                use_container_width=True,
                key="splits_editor",
            )

            with st.expander("Inspect a Specific Split in CAVE + Neuroglancer (3D)"):
                row_idx = st.selectbox(
                    "Select split row to preview",
                    options=edited_splits.index.tolist(),
                    format_func=lambda i: (
                        f"Root {edited_splits.loc[i, 'target_root_id']} | "
                        f"Source skid {edited_splits.loc[i, 'source_skid']} "
                        f"({edited_splits.loc[i, 'source_nodes']} nodes)"
                    ),
                    key="preview_split_row_selector",
                )
                if st.button("Preview Selected Split", key="btn_preview_single_split"):
                    try:
                        cave_client, _ = init_clients_from_project(config)
                        row = edited_splits.loc[row_idx]
                        res = preview_single_split(
                            st.session_state["mapped_df"],
                            cave_client,
                            root_id=int(row["target_root_id"]),
                            source_skid=int(row["source_skid"]),
                            sink_skids=[int(x) for x in str(row["sink_skids"]).split(",")],
                            cave_resolution_nm=tuple(config["cave_resolution_nm"]),
                        )
                        st.write(
                            f"Split valid: **{res['split_valid']}** "
                            f"({res['predicted_component_count']} predicted components)"
                        )
                        st.markdown(f"[Open 3D Cut Preview in Neuroglancer]({res['neuroglancer_url']})")
                    except Exception as exc:
                        st.error(f"Preview failed: {exc}")
        else:
            edited_splits = splits_df
            st.info("No false merges detected (0 splits needed).")

        if not merges_df.empty:
            st.markdown("**Review & Approve Merges:**")
            edited_merges = st.data_editor(
                merges_df,
                use_container_width=True,
                key="merges_editor",
            )
        else:
            edited_merges = merges_df
            st.info("No false splits detected (0 merges needed).")

        # 4. Execute Approved Batch
        has_approved_splits = not edited_splits.empty and bool(edited_splits["approve"].any())
        has_approved_merges = not edited_merges.empty and bool(edited_merges["approve"].any())

        if has_approved_splits or has_approved_merges:
            if st.button("Execute Approved Splits & Merges in CAVE", type="primary", key="btn_execute_batch"):
                cave_client, _ = init_clients_from_project(config)
                bar = st.progress(0.0)
                status_txt = st.empty()

                def _cb(step: int, total: int, msg: str):
                    bar.progress(step / max(total, 1))
                    status_txt.text(f"[{step}/{total}] {msg}")

                audit_path = Path(f"proofreading_audit_log_{project_name}.csv")
                audit_df = execute_approved_edits(
                    st.session_state["mapped_df"],
                    edited_splits,
                    edited_merges,
                    cave_client,
                    cave_resolution_nm=tuple(config["cave_resolution_nm"]),
                    audit_log_path=audit_path,
                    progress_callback=_cb,
                )
                st.success(f"Batch finished! Audit log saved to `{audit_path}`")
                st.dataframe(audit_df, use_container_width=True)


if __name__ == "__main__":
    render_catmaid_proofreading_tab()