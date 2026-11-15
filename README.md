# Developers
Brewed by Michael Clayton
Spiced by Samia Mohinta

# Segment splitter

A small web app that cuts large, fused CAVE segments into smaller pieces, to make manual proofreading easier. Each split is a **real chunkedgraph edit**.

| File | Purpose |
|---|---|
| `app.py` | Web server and page (Python standard library only) |
| `splitter.py` | Chooses terminals, previews, and executes one cut |
| `config.json` | The samples in the menu and the CAVE server |
| `diagnose.py` | Read-only check of which terminal layouts the server accepts for one segment |
| `end_user/guide.html` | User guide with screenshots, also shown in the app under **How to use** |
| `requirements.txt` | Python packages (`pip install -r requirements.txt`) |

## Run

Needs Python 3.9+ and the packages in `requirements.txt`.

```
/lmb/home/mclayton/venvs/connectomics-env/bin/python app.py --selftest   # fake server, no CAVE calls
/lmb/home/mclayton/venvs/connectomics-env/bin/python app.py              # prints a link containing the access token
```

To use a config file kept elsewhere, add `--config /path/to/config.json`.

The app listens on `localhost:8765` (change with `--port`). To open it from your own machine, run `ssh -L 8765:localhost:8765 <host>`, then open the printed link. Start it in `screen` or `tmux` to keep it running. It needs CAVE credentials set up on the host, and each start prints a new link.

## CAVE token

The app uses the CAVE token saved for the Linux user that runs it. When it is missing or expired, jobs fail with "401 … Token is Invalid or Expired" and nothing is edited. To renew it, run these on the server with the app's Python, as that user:

```
python -c "from caveclient import CAVEclient; CAVEclient(server_address='https://global.connectomics.braininbrain.org').auth.get_new_token(open=False)"
python -c "from caveclient import CAVEclient; CAVEclient(server_address='https://global.connectomics.braininbrain.org').auth.save_token(token='PASTE_TOKEN_HERE', overwrite=True)"
```

The first prints a URL: open it, log in, and copy the token into the second. No restart is needed. If a job was marked "Outcome unknown" because of this, click **Re-check** once the token works.

## Use

1. Choose a sample. To add one, add a line to `"samples"` in `config.json` (menu name → datastack name), then restart the app:

   ```json
   {
     "server_address": "https://global.connectomics.braininbrain.org",
     "samples": {
       "Parker-Goog": "zlatic_parker_google_segmentation_datastack",
       "Parker_LSD": "zlatic_parker_v1b_datastack"
     }
   }
   ```

   The app refuses to start, and says why, if the file is missing or malformed.
2. Enter one segment ID and click **Start**.
   - **Preview first** (default) runs a read-only preview, then makes the cut without asking if it passes. A failed preview edits nothing.
   - Unticked, it goes straight to the real edit after a confirmation prompt.
3. Set **Rounds** to keep cutting automatically (up to 6). Each round cuts the two large pieces from the previous one, so A → B + C, then B → B0 + B1 and C → C0 + C1. N rounds is at most 2^N − 1 edits.
   - With **Preview first**, every cut, including the first, runs its own preview and splits without asking if that passes.
   - Unticked, you confirm once and no cut in the tree is previewed.
4. Follow the jobs under **Progress**, where each piece is nested under its parent. A finished split lists its op number and new root IDs. **Copy** next to a result copies it with the sample, segment, status and, for errors, the full server response (the page shows only the server's message).

A branch stops, while the others continue, if its preview fails, its cut can't be verified or doesn't separate the sides, or its outcome is unknown. Pieces under 10 L2 nodes and small fragments aren't cut. **Re-check** doesn't resume a stopped branch; paste a new ID in to continue it.

To stop a running tree, stop the app: running edits still finish, and queued jobs are marked failed on restart.

Each cut gives two large pieces and sometimes a few small fragments. Previews don't show those fragments.

## How a cut works

1. Read the root's L2 graph and bisect it spectrally.
2. Grow a connected ball (20% of nodes) from the deepest node of each half, and take about 40 spread points from each ball as terminals.
3. Add the outermost L2 node on each axis to its nearest side. Without these, the sides can stay joined around the box the terminals span (op 1581 failed this way; op 1594 worked with them).
4. Prefer L2 nodes with 2+ supervoxels as terminals. A lone supervoxel only has edges across chunk borders, which can fall outside the server's cut graph ("Invalid vertex index").
5. Send the terminals to `preview_split` or `execute_split`. If the preview fails with "Invalid vertex index" or "paths … irreparably overlap", it retries up to 3 times: first without the outermost nodes (overlap only), then with a random half of the terminals. Execute never retries, and uses the exact terminals of the passed preview, or fresh ones if there was no preview.

A preview passes if the split is legal, every terminal is on its own side, at most 1% of supervoxels are missing, no L2 links remain between the sides, and the smaller side has at least 10% of L2 nodes. Passing doesn't guarantee a real split, but it has predicted the sizes well so far.

## Practical notes

- **Edits are permanent, but can be undone:** use `cg.undo_operation(op)`, newest first. Root IDs change with every edit, so anything keyed on an old root is stale.
- **Don't kill a running edit:** the server still applies it, and the log row is lost. Stopping the app doesn't cancel it either.
- **"Outcome unknown":** the request failed, but the edit may have applied. The segment is blocked until you click **Re-check**.
- **Skipping the preview:** a failed cut only shows up after the edit, so keep the preview on for large segments.
- **Limits:** segments under 10 L2 nodes are refused, and up to 4 jobs run at once across all trees. Time per cut grows with segment size, from about 15 s at a few hundred L2 to about 3 min at 6,000 L2.
- **Access:** the token in the link allows edits with your CAVE credentials, so don't share it. Using `--host` for anything but localhost sends it unencrypted.

## Outputs

Written to `outputs/`, which isn't gitignored:

- `gui_jobs.json`: progress, reloaded on restart
- `<root>/edit_log.csv`: one row per edit, with no header: op, root, new roots, L2 sizes, source and sink terminals
- `<root>/preview_terminals.json`: the latest terminals for that root
