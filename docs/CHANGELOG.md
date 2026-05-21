# Changelog

## 0.4 — In progress

### Major features
- **Roster + manual track-assignment** (new `web/pages/3_Roster.py`). Define
  team rosters once, drag tracks onto roster slots, and the Viewer's
  Overview + Tracks + radar all aggregate per-player rather than per-track.
  Unassigned tracks are grouped together so nothing is lost. Saved as
  `<out>/roster.json`. The Viewer sidebar exposes an "Aggregate by player"
  toggle that flips every tab between per-track and per-player views.
- **Synced replay** in the Viewer's Replay tab. New checkbox "Sync camera
  video to the timeline" puts the camera video side-by-side with the
  tactical board; scrubbing the slider re-seeks the HTML5 video element.
  Requires an H.264 `camera.mp4` (now the default).

### Inference performance
- `--imgsz N` CLI flag and Analyse-page slider for YOLO inference resolution.
  Default `640` (was native source size).
- `--frame-stride N` CLI flag and Analyse-page input. Direct 1/N throughput
  multiplier; KPI distances remain correct.
- `import supervision` hoisted out of the per-frame hot loop.
- `FieldValidator.filter_detections` vectorised — single polygon test per
  surviving detection instead of two.

### Viewer UX
- **"All players" heatmap mode** as the new default (alongside per-team and
  per-track).
- **Replay plays at the source video's effective fps**, no longer hardcoded
  to ~20. New playback-speed select slider (0.25× → 8×).
- **Radar chart uses percentile-based normalisation** across the squad, so
  axes like "Seen" no longer dominate the chart.
- Calibration-drift warning at the top of the Heatmaps tab when > 0 points
  fall outside the rendered pitch.
- Per-tab labels switch between "Track" and "Player" automatically based
  on the roster toggle.

### Analyser robustness
- **Run-lock kill button** in the Analyse page's red "Another analyser is
  already running" banner. Sends SIGTERM, falls back to SIGKILL after 2 s,
  then cleans up the lock file.
- **Stream re-open** on persistent `cap.read()` failure. Helps live streams
  that lose their CDN URL mid-match (~30 consecutive failed reads triggers
  one `open_youtube_stream` retry before giving up).
- **Early calibration validation** in the analyser: after ~200 frames, logs
  a `[CALIBRATION]` warning if > 30% of mapped positions fall outside the
  board. Saves hours of wall time on bad-calibration runs.
- **Browser-friendly video codec**. `_open_browser_friendly_writer` tries
  `avc1` → `H264` → `mp4v`; the Viewer's Video tab classifies each MP4 and
  shows a clear error for `mp4v`-only or `moov`-less (killed-mid-run) files,
  plus a Download button regardless.
- **KPI CSV flushes every 30 frames** (atomically) so the Viewer's Overview
  tab populates progressively instead of being empty until the run ends.
- **Speed cap tightened** from 12 m/s to 10 m/s (the world-record 100 m
  sprint is ~10.4 m/s, so anything above this is almost certainly an
  ID-swap teleport).

### Analyse page polish
- **Local video upload** via `st.file_uploader` alongside the YouTube URL
  input. Saved to `<out>/_source.<ext>`; old `_source.*` files are
  auto-cleaned when a new upload arrives.
- **Start-time field accepts HH:MM:SS** (was MM:SS only). Now matches the
  fixed `parse_start_time`.
- **Live preview every N frames** with a default of 2 (was 10) and a
  downscale to 960 px wide → fewer "stoppy" jumps.
- **"Reset session state" expander** clears the cached VideoCapture, queued
  subprocess, log lines, and points so a fresh page is one click away.

### Legacy
- The OpenCV calibration UI is now flagged as deprecated in its docstring;
  the CLI logs a tip pointing users to the browser calibrator. The OpenCV
  path still works for headless servers.

### Testing & CI
- **118 tests** (up from 77). New `tests/test_runtime_helpers.py`,
  `tests/test_stream.py`, `tests/test_web_shared.py` (now also covering
  `save_roster` / `load_roster` / `player_label` / `apply_roster_to_kpis`).
- CI matrix: `ubuntu-latest` + `windows-latest`, Python 3.10/3.11/3.12.
- `ruff check src tests web` runs as a separate `lint` job.
- CI installs `[dev,viewer]` together to catch Streamlit import regressions.

### Dependencies
- `streamlit-image-coordinates>=0.1.6` for click-to-place calibration.
- `supervision>=0.25,<0.30` pinned until we migrate off the deprecated
  `sv.ByteTrack` alias.

### Configuration cleanup
- Removed dead `Config` fields: `min_players_for_train`, `smoothing_window`,
  `duel_radius_px`, `possession_radius_px`, `track_activation_threshold`.

---

## 0.3 — Bug fix sprint

### Robustness
- `run()` in `__main__.py` now claims the run-lock as the first owned resource
  and consolidates all cleanup in a single try/finally — error returns
  (model load, video writer creation, etc.) no longer leak the lock file or
  truncate the MP4 / JSONL outputs.
- `_write_snapshot` writes to `name.tmp.ext` (suffix preserved) so OpenCV can
  pick the right encoder. Previous code created `name.ext.tmp` which silently
  failed every JPEG encode.
- `web/_shared.py::load_positions` skips malformed JSONL lines instead of
  crashing the viewer, and reports the skip count back to the UI.
- `web/_shared.py::load_kpis` uses `on_bad_lines="skip"` to tolerate partial
  CSV rows.
- `styled_plot` merges the default Plotly layout dict with caller-provided
  overrides instead of double-passing kwargs — fixes the `margin` collision
  that broke the Replay tab.
- `SimpleFieldMapper.transform` returns `None` on failure rather than `(0, 0)`,
  preventing phantom hot spots in the top-left corner of every heatmap.
- `parse_start_time` now accepts `HH:MM:SS` as well as `MM:SS` / `SS`.

### Features
- **Two-page Streamlit app** (`web/app.py` landing + `web/pages/1_Analyse.py`
  + `web/pages/2_Viewer.py`):
  - Analyse page accepts a YouTube URL *or* a locally uploaded video.
  - In-browser 6-point calibration with click-to-place + per-point re-place
    + adjustable "click margin" so points can be placed outside the camera frame.
  - Launches the CLI as a subprocess with `--no-gui`; live log + per-second
    Annotated Camera + Tactical Board previews.
- Live snapshot pipeline: analyser writes `_live_camera.jpg` / `_live_board.jpg`
  atomically every N frames (`--snapshot-every`, `--snapshot-dir`,
  `--snapshot-width`). Snapshots are downscaled (default 960 px wide) to keep
  the browser-refresh loop fast.
- **Run-lock**: analyser writes `<out>/.run.lock` on startup, touches it every
  30 frames, and removes it on exit. Analyse page refuses to spawn a second
  subprocess against the same directory while a fresh lock is present.

### Per-track honesty in the Viewer
- "Players" tab renamed to **Tracks** with a sidebar "Track filter" slider that
  hides fragments shorter than the chosen duration.
- Header subtitle and copy reworded throughout to say "track" instead of
  "player" where appropriate.
- Information banner explains that motion-only tracking (ByteTrack) starts a
  fresh ID whenever a player is occluded, leaves the frame, or is re-acquired.
- Team-level totals on the Overview tab are unaffected and remain accurate.

### Testing & CI
- Test count: **107** (up from 77). New coverage:
  - `_write_snapshot` (atomic rename, downscaling, no leftover `.tmp` files).
  - `_claim_lock` / `_lock_is_active` / `_refresh_lock`.
  - `parse_start_time` (incl. the new `HH:MM:SS` case).
  - `open_youtube_stream` local-file fast path.
  - `web/_shared.load_positions` / `load_kpis` robustness against
    malformed inputs.
  - 6-point homography numerical accuracy at every calibration vertex.
- CI matrix expanded to `windows-latest` + `ubuntu-latest`, Python 3.10/3.11/3.12.
- New `lint` job runs `ruff check` over `src` / `tests` / `web`.
- CI installs `[viewer]` extras alongside `[dev]` so Streamlit imports are
  exercised on every push.

### Config cleanup
- Removed dead `Config` fields that nothing read: `min_players_for_train`,
  `smoothing_window`, `duel_radius_px`, `possession_radius_px`,
  `track_activation_threshold`. The KPI thresholds live on `KPITracker`
  where they are actually consumed.

### Theming
- Streamlit theme is dark with a muted rose-pink (`#d85c7d`) accent matching
  the Plotly chart colours. `.streamlit/config.toml` plus inline CSS in
  `web/_shared.py::CUSTOM_CSS`.

### Dependencies
- Pinned `supervision>=0.25,<0.30` until we migrate off the soon-to-be-removed
  top-level `sv.ByteTrack` alias.
- Added `streamlit-image-coordinates>=0.1.6` to the `[viewer]` extras for
  click-to-place calibration.

---

## 0.2 — Repository restructure

- Moved all application code into `src/futsal_analytics/` package.
- Split `futsal_analyzer.py` into focused modules:
  `config`, `stream`, `calibration`, `field`, `detection`, `board`,
  `kpis`, `__main__`.
- Replaced duplicated `FieldCalibrator` in `calibracion_simple.py` with a
  shared import from `futsal_analytics.calibration`.
- Added `pyproject.toml` with declared dependencies and console-script entry
  points (`futsal-analytics`, `futsal-calibrate`).
- Added `pytest`-based test suite under `tests/`.
- Translated all documentation to English; moved docs to `docs/`.
- Added `argparse` CLI with `--no-gui`, `--save-{video,board-video,positions,kpis,calibration}`,
  `--device`, `--max-frames`, `--retrain-every`, `--skip-when-behind`,
  `--allow-frame-pick`.
- Integrated `supervision.ByteTrack` for persistent player IDs across frames.
- Added `KPITracker` (distance, top speed, sprint count, possession, duels) and
  `PositionLogger` (per-frame JSONL).
- Added FIFA-spec futsal pitch markings on the `TacticalBoard`.
- 6-point homography (`cv2.findHomography`) using halfway-line points as
  additional correspondences (more accurate than the 4-corner
  `getPerspectiveTransform`).

---

## 0.1 — Calibration evolution

### 4-point → 6-point polygon
- **Problem**: Original calibrator used only 4 corner points, providing no
  information about the halfway line.
- **Change**: Added two additional control points — CT (Centre-Top) and CB
  (Centre-Bottom) — which mark the intersection of the halfway line with the
  top and bottom pitch edges.
- **Point order** (current): `TL, CT, TR, BR, CB, BL`.

### Mouse interaction fix
- **Problem**: Mouse click events were being dropped because the drag state was
  stored in local variables inside the callback closure, which were not shared
  correctly across events.
- **Fix**: Moved `dragging` and `active_point` to instance attributes.

### FieldValidator: polygon-based validation
- Switched from a simple bounding-box check to `cv2.pointPolygonTest` against
  the full 6-point polygon. A detection is kept when either the foot position
  or the bounding-box centre lies within the polygon.

### Lazy ML imports
- `ultralytics` (YOLO) and `scikit-learn` (KMeans) are now imported only when
  `setup_detectors` or `TeamClassifier.__init__` is first called, so the
  calibration UI opens quickly without loading the full ML stack.
