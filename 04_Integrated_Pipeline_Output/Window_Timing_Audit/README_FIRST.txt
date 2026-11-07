CAPPIMU 50-ROW WINDOW TIMING AUDIT (MATCHES CURRENT PIPELINE)
=============================================================

The script mirrors the project's integrated raw pipeline segmentation:
- recording split: label change, dt < 0, or dt > 500 ms
- inner feature-segment split: label change, dt < 0, or dt > 100 ms
- dt == 0 rows are preserved
- complete non-overlapping 50-row windows are measured

Files: subject summary, all-window timing CSV, segment inventory, missing/error inventory, JSON summary.
Check that total_complete_50row_windows equals 110,949. No source CSV or model is modified.
