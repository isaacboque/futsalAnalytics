"""
Analyse page — end-to-end web flow:

    1. Enter a YouTube URL + start time + output directory.
    2. Fetch a still frame (skip forward in time to find a clean one).
    3. Place 6 calibration points on the frame (TL, CT, TR, BR, CB, BL).
    4. Launch the ``futsal-analytics`` CLI as a subprocess with --no-gui.
    5. Stream stdout into a live log panel while the analyser runs.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `_shared` import

from _shared import inject_css, render_sidebar_brand

from futsal_analytics.calibration import save_calibration
from futsal_analytics.config import Config
from futsal_analytics.stream import (
    open_youtube_stream,
    parse_start_time,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Analyse · Futsal Analytics",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
render_sidebar_brand()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


POINT_LABELS = ["TL", "CT", "TR", "BR", "CB", "BL"]
POINT_DESCS = {
    "TL": "Top-Left corner",
    "CT": "Centre-Top (halfway line, top edge)",
    "TR": "Top-Right corner",
    "BR": "Bottom-Right corner",
    "CB": "Centre-Bottom (halfway line, bottom edge)",
    "BL": "Bottom-Left corner",
}
# Display width for the calibration image (px); height is computed from the
# frame aspect ratio. Smaller than 1080p so the whole pitch fits on screen.
CAL_DISPLAY_W = 960

PROGRESS_RE = re.compile(r"Frame (\d+)\s+\|\s+wall\s+([\d.]+)s")


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("an_source_kind", "youtube")         # "youtube" or "local"
    ss.setdefault("an_url", "")
    ss.setdefault("an_local_path", "")                 # absolute path of uploaded MP4
    ss.setdefault("an_start", "0:00")
    ss.setdefault("an_out_dir", str(Path("out").absolute()))
    ss.setdefault("an_cap", None)                      # cv2.VideoCapture or None
    ss.setdefault("an_frame", None)                    # np.ndarray BGR or None
    ss.setdefault("an_points", [])                     # list[tuple[float, float]]
    ss.setdefault("an_next_point", 0)                  # 0..5
    ss.setdefault("an_replace_idx", None)              # int or None
    ss.setdefault("an_proc", None)                     # subprocess.Popen or None
    ss.setdefault("an_log_lines", [])                  # list[str]
    ss.setdefault("an_log_queue", None)                # queue.Queue or None
    ss.setdefault("an_reader_thread", None)            # threading.Thread or None
    ss.setdefault("an_frames_done", 0)
    ss.setdefault("an_started_at", 0.0)
    ss.setdefault("an_finished_at", 0.0)


_init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


st.markdown(
    '<div class="fa-hero"><h1>Analyse a match</h1>'
    '<span class="tag">URL \u2192 calibrate \u2192 run</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="fa-meta">Everything happens in the browser. No PowerShell required.</p>',
    unsafe_allow_html=True,
)

with st.expander("Reset this page", expanded=False):
    st.caption(
        "Clears the URL / uploaded file / calibration / queued subprocess "
        "from the in-memory session. Disk artefacts (`cal.npy`, recorded "
        "videos, KPIs) are kept."
    )
    if st.button("Reset session state", use_container_width=False):
        # Release the cached VideoCapture before dropping the reference.
        old_cap = st.session_state.get("an_cap")
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass
        # Drop every key starting with 'an_' so the next render re-initialises.
        for k in list(st.session_state.keys()):
            if k.startswith("an_") or k.startswith("_an_"):
                del st.session_state[k]
        st.toast("Session reset.", icon=":material/refresh:")
        st.rerun()


# ---------------------------------------------------------------------------
# Step 1 — Source
# ---------------------------------------------------------------------------


st.markdown("### 1\u2002\u00b7\u2002Source")

source_kind = st.radio(
    "Source",
    options=["YouTube URL", "Local file"],
    index=0 if st.session_state["an_source_kind"] == "youtube" else 1,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["an_source_kind"] = "youtube" if source_kind == "YouTube URL" else "local"

if st.session_state["an_source_kind"] == "youtube":
    col_url, col_start, col_out = st.columns([3, 1, 2])
    with col_url:
        st.session_state["an_url"] = st.text_input(
            "YouTube URL",
            value=st.session_state["an_url"],
            placeholder="https://www.youtube.com/live/...",
        )
    with col_start:
        st.session_state["an_start"] = st.text_input(
            "Start (HH:MM:SS)",
            value=st.session_state["an_start"],
        )
    with col_out:
        st.session_state["an_out_dir"] = st.text_input(
            "Output directory",
            value=st.session_state["an_out_dir"],
            help="Where positions.jsonl, kpis.csv, cal.npy and videos will be written.",
        )
else:
    col_up, col_start, col_out = st.columns([3, 1, 2])
    with col_up:
        uploaded = st.file_uploader(
            "Local video file",
            type=["mp4", "mov", "mkv", "avi"],
            help="Stored temporarily under the output directory as `_source.<ext>`.",
        )
        if uploaded is not None:
            out_dir_tmp = Path(st.session_state["an_out_dir"])
            out_dir_tmp.mkdir(parents=True, exist_ok=True)
            ext = Path(uploaded.name).suffix or ".mp4"
            target = out_dir_tmp / f"_source{ext}"
            existing_size = target.stat().st_size if target.exists() else -1
            if existing_size != uploaded.size:
                # Sweep any previous _source.* with a different extension so we
                # don't accumulate gigabytes of old uploads in the output dir.
                for stale in out_dir_tmp.glob("_source.*"):
                    if stale != target:
                        try:
                            stale.unlink()
                        except OSError:
                            pass
                with open(target, "wb") as fp:
                    fp.write(uploaded.getbuffer())
                st.toast(f"Saved {target.name} ({uploaded.size/1_048_576:.1f} MB)",
                         icon=":material/save:")
                # Force re-open on next click
                st.session_state["an_cap"] = None
                st.session_state["an_frame"] = None
            st.session_state["an_local_path"] = str(target)
        if st.session_state["an_local_path"]:
            st.caption(f"Using `{st.session_state['an_local_path']}`")
    with col_start:
        st.session_state["an_start"] = st.text_input(
            "Start (HH:MM:SS)",
            value=st.session_state["an_start"],
            key="an_start_local",
        )
    with col_out:
        st.session_state["an_out_dir"] = st.text_input(
            "Output directory",
            value=st.session_state["an_out_dir"],
            help="Where positions.jsonl, kpis.csv, cal.npy and videos will be written.",
            key="an_out_dir_local",
        )

out_dir = Path(st.session_state["an_out_dir"])


# ---------------------------------------------------------------------------
# Step 2 — Calibration frame picker
# ---------------------------------------------------------------------------


st.markdown("### 2\u2002\u00b7\u2002Pick a calibration frame")

cap = st.session_state.get("an_cap")


def _open_stream() -> Optional[cv2.VideoCapture]:
    if st.session_state["an_source_kind"] == "youtube":
        if not st.session_state["an_url"].strip():
            st.error("Enter a YouTube URL first.")
            return None
        cfg = Config()
        with st.spinner("Opening stream\u2026"):
            new_cap = open_youtube_stream(st.session_state["an_url"], cfg)
        if new_cap is None or not new_cap.isOpened():
            st.error("Could not open the stream. Check the URL and try again.")
            return None
    else:
        path = st.session_state["an_local_path"]
        if not path or not Path(path).exists():
            st.error("Upload a video file first.")
            return None
        new_cap = cv2.VideoCapture(path)
        if not new_cap.isOpened():
            st.error(f"OpenCV could not open `{path}`.")
            return None

    start_s = parse_start_time(st.session_state["an_start"])
    if start_s > 0:
        new_cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    return new_cap


def _read_next(cap_: cv2.VideoCapture, n: int = 1) -> Optional[np.ndarray]:
    """Read forward *n* frames; return the last one (drops the rest)."""
    frame = None
    for _ in range(max(1, n)):
        ret, f = cap_.read()
        if not ret or f is None:
            break
        frame = f
    return frame


col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 2])
with col_btn1:
    if st.button("Open stream", use_container_width=True, type="primary"):
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        new_cap = _open_stream()
        if new_cap is not None:
            st.session_state["an_cap"] = new_cap
            fps = new_cap.get(cv2.CAP_PROP_FPS) or 25.0
            st.session_state["an_fps"] = fps
            st.session_state["an_frame"] = _read_next(new_cap, 1)
            st.session_state["an_points"] = []
            st.session_state["an_next_point"] = 0
            st.rerun()

cap = st.session_state.get("an_cap")
fps = float(st.session_state.get("an_fps", 25.0))

with col_btn2:
    next_disabled = cap is None
    if st.button("Next frame", use_container_width=True, disabled=next_disabled):
        f = _read_next(cap, 1)
        if f is not None:
            st.session_state["an_frame"] = f
            st.rerun()
        else:
            st.warning("End of stream reached.")
with col_btn3:
    skip30_disabled = cap is None
    if st.button("Skip 30 s", use_container_width=True, disabled=skip30_disabled):
        f = _read_next(cap, int(30 * fps))
        if f is not None:
            st.session_state["an_frame"] = f
            st.rerun()
        else:
            st.warning("End of stream reached.")
with col_btn4:
    skip60_disabled = cap is None
    if st.button("Skip 60 s", use_container_width=True, disabled=skip60_disabled):
        f = _read_next(cap, int(60 * fps))
        if f is not None:
            st.session_state["an_frame"] = f
            st.rerun()
        else:
            st.warning("End of stream reached.")


frame = st.session_state.get("an_frame")


# ---------------------------------------------------------------------------
# Step 3 — Place the 6 calibration points
# ---------------------------------------------------------------------------


def _draw_overlay(
    frame_bgr: np.ndarray,
    points: List[tuple],
    padding: int = 0,
) -> np.ndarray:
    """Return a padded canvas with the frame, polygon, and numbered handles.

    Points are stored in *frame* coordinates (origin at the frame's top-left).
    They are translated by ``padding`` when drawn so handles outside the frame
    still render inside the larger click canvas.
    """
    fh, fw = frame_bgr.shape[:2]
    canvas = np.full((fh + 2 * padding, fw + 2 * padding, 3),
                     (32, 32, 36), dtype=np.uint8)
    canvas[padding : padding + fh, padding : padding + fw] = frame_bgr

    if padding > 0:
        cv2.rectangle(
            canvas,
            (padding, padding),
            (padding + fw - 1, padding + fh - 1),
            (160, 160, 170),
            2,
        )

    cpts = [(x + padding, y + padding) for x, y in points]

    if len(cpts) >= 2:
        for i in range(len(cpts)):
            if i + 1 < len(cpts) or len(cpts) == 6:
                a = tuple(map(int, cpts[i]))
                b = tuple(map(int, cpts[(i + 1) % len(cpts)]))
                cv2.line(canvas, a, b, (240, 240, 240), 2, cv2.LINE_AA)

    if len(cpts) == 6:
        overlay = canvas.copy()
        poly = np.array([[int(x), int(y)] for x, y in cpts], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (220, 130, 50))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    for idx, (x, y) in enumerate(cpts):
        ix, iy = int(x), int(y)
        cv2.circle(canvas, (ix, iy), 14, (0, 145, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (ix, iy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (ix - 5, iy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, POINT_LABELS[idx], (ix - 18, iy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


st.markdown("### 3\u2002\u00b7\u2002Calibrate")

if frame is None:
    st.info("Open a stream above and pick a frame, then place the 6 points here.")
else:
    fh, fw = frame.shape[:2]

    pad_pct = st.slider(
        "Click margin around the frame (%)",
        min_value=0, max_value=60,
        value=int(st.session_state.get("an_pad_pct", 25)),
        step=5,
        help=(
            "Extra clickable space outside the video frame. Increase this if a "
            "pitch corner is cropped by the camera and you need to place a point "
            "off-screen."
        ),
        key="an_pad_pct",
    )
    padding = int(fh * pad_pct / 100)

    canvas_w = fw + 2 * padding
    canvas_h = fh + 2 * padding
    display_w = min(CAL_DISPLAY_W, canvas_w)
    scale = display_w / canvas_w
    display_h = int(canvas_h * scale)

    overlay_bgr = _draw_overlay(frame, st.session_state["an_points"], padding)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay_resized = cv2.resize(overlay_rgb, (display_w, display_h),
                                 interpolation=cv2.INTER_AREA)

    points = st.session_state["an_points"]
    next_idx = st.session_state["an_next_point"]
    replace_idx = st.session_state.get("an_replace_idx")

    if replace_idx is not None:
        st.info(
            f"Click anywhere on the image to **move point {replace_idx} "
            f"({POINT_LABELS[replace_idx]} \u2014 {POINT_DESCS[POINT_LABELS[replace_idx]]})**.",
            icon=":material/touch_app:",
        )
    elif len(points) < 6:
        st.info(
            f"Click to place point **{next_idx} ({POINT_LABELS[next_idx]} \u2014 "
            f"{POINT_DESCS[POINT_LABELS[next_idx]]})**.",
            icon=":material/touch_app:",
        )
    else:
        st.success(
            "All 6 points placed. Use the buttons below to re-place any point, "
            "or save the calibration.",
            icon=":material/check_circle:",
        )

    col_img, col_pts = st.columns([4, 1])
    with col_img:
        click = streamlit_image_coordinates(
            overlay_resized,
            key=f"an_click_{len(points)}_{replace_idx}_{padding}",
        )
        if click is not None:
            # Display \u2192 canvas \u2192 frame (subtract padding so off-frame clicks
            # become negative or > fw/fh, which is fine for cv2.findHomography).
            cx_canvas = float(click["x"]) / scale
            cy_canvas = float(click["y"]) / scale
            cx = cx_canvas - padding
            cy = cy_canvas - padding
            if replace_idx is not None:
                st.session_state["an_points"][replace_idx] = (cx, cy)
                st.session_state["an_replace_idx"] = None
                st.rerun()
            elif len(points) < 6:
                st.session_state["an_points"].append((cx, cy))
                st.session_state["an_next_point"] = len(st.session_state["an_points"]) % 6
                st.rerun()
        if padding > 0:
            st.caption(
                f"Clickable area extends {padding}\u202fpx beyond the video frame "
                f"on each side."
            )

    with col_pts:
        st.markdown("**Points**")
        for i in range(6):
            placed = i < len(points)
            label = POINT_LABELS[i]
            if placed:
                px_, py_ = points[i]
                txt = f"{i} {label}  ({int(px_)}, {int(py_)})"
            else:
                txt = f"{i} {label}  —"
            if st.button(
                txt, key=f"an_re_{i}",
                use_container_width=True,
                disabled=not placed and i != st.session_state["an_next_point"],
            ):
                if placed:
                    st.session_state["an_replace_idx"] = i
                    st.rerun()

        if st.button("Reset all", use_container_width=True):
            st.session_state["an_points"] = []
            st.session_state["an_next_point"] = 0
            st.session_state["an_replace_idx"] = None
            st.rerun()

        save_disabled = len(points) < 6
        if st.button(
            "Save calibration",
            use_container_width=True,
            type="primary",
            disabled=save_disabled,
        ):
            out_dir.mkdir(parents=True, exist_ok=True)
            cal_path = out_dir / "cal.npy"
            save_calibration(np.asarray(points, dtype=np.float32), cal_path)
            st.toast(f"Saved {cal_path}", icon=":material/save:")
            st.session_state["an_cal_path"] = str(cal_path)


# ---------------------------------------------------------------------------
# Step 4 — Run the analyser
# ---------------------------------------------------------------------------


st.markdown("### 4\u2002\u00b7\u2002Run")

cal_path = st.session_state.get("an_cal_path") or str(out_dir / "cal.npy")
cal_ready = Path(cal_path).exists()

if not cal_ready:
    st.info(
        "Save a calibration above (or place an existing `cal.npy` in the output "
        "directory) before starting the analysis.",
        icon=":material/info:",
    )

col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    rec_camera = st.checkbox("Record annotated camera (camera.mp4)", value=True)
    rec_board = st.checkbox("Record tactical board (board.mp4)", value=True)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    retrain_every = st.number_input(
        "Retrain classifier every N frames (0 = never)",
        min_value=0, max_value=10_000, value=600, step=100,
    )
with col_opts2:
    imgsz = st.select_slider(
        "YOLO inference size (px)",
        options=[320, 416, 480, 544, 640, 768, 960],
        value=640,
        help=(
            "Lower = faster but may miss small/far players. 320 is roughly 4\u00d7 "
            "faster than 960 on CPU."
        ),
    )
    frame_stride = st.number_input(
        "Frame stride (process every Nth frame)",
        min_value=1, max_value=15, value=1, step=1,
        help=(
            "Stride 2 doubles throughput at the cost of halving temporal "
            "resolution. KPI distances stay correct."
        ),
    )
    snapshot_every = st.number_input(
        "Live preview every N frames (0 = off)",
        min_value=0, max_value=300, value=2, step=1,
        help=(
            "How often the analyser dumps a fresh preview JPEG. 1 = every frame "
            "(smoothest, ~10\u202fms extra per frame). 0 disables previews."
        ),
    )

with st.expander("Tracking quality (advanced)"):
    st.caption(
        "ByteTrack uses motion + IoU only \u2014 no appearance features. The "
        "single biggest lever is **lost-track buffer**: how long to remember "
        "a track after detection drops. Higher = fewer ID switches across "
        "occlusions, but stale tracks may linger."
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tracker_buffer = st.slider(
            "Lost-track buffer (frames)",
            min_value=15, max_value=300, value=90, step=15,
            help="Default 90 \u2248 3 s @ 30 fps. Try 150 if players keep "
                 "losing their ID during occlusions.",
        )
        tracker_minframes = st.slider(
            "Min consecutive frames before activation",
            min_value=1, max_value=10, value=2, step=1,
            help="Higher kills 1-frame flicker detections that would spawn "
                 "ghost tracks. 2-3 is a good range for futsal.",
        )
    with col_t2:
        tracker_conf = st.slider(
            "Track activation confidence",
            min_value=0.10, max_value=0.70, value=0.35, step=0.05,
            help="Detection confidence required to start a track. Higher "
                 "filters more false positives.",
        )
        tracker_iou = st.slider(
            "Minimum matching IoU",
            min_value=0.30, max_value=0.95, value=0.70, step=0.05,
            help="Lower allows faster apparent motion between frames "
                 "(broadcast pan / camera shake).",
        )

col_run, col_stop = st.columns([1, 1])

proc: Optional[subprocess.Popen] = st.session_state.get("an_proc")
running = proc is not None and proc.poll() is None

# Detect a foreign analyser running against the same output dir (e.g.
# orphaned from a browser refresh that lost our session_state reference).
LOCK_STALE_S = 10.0
lock_path = out_dir / ".run.lock"
foreign_running = False
foreign_age = 0.0
foreign_info = ""
if lock_path.exists():
    try:
        foreign_age = time.time() - lock_path.stat().st_mtime
        if foreign_age < LOCK_STALE_S:
            foreign_info = lock_path.read_text(encoding="utf-8")
            foreign_running = not running  # only "foreign" if it's not our own
    except OSError:
        pass

if foreign_running:
    st.error(
        f"Another analyser is already running in `{out_dir}` "
        f"(lock {foreign_age:.1f}s old).\n\n"
        f"```\n{foreign_info}\n```",
        icon=":material/lock:",
    )

    # Try to extract the foreign PID so the kill button knows what to terminate.
    foreign_pid: Optional[int] = None
    for line in foreign_info.splitlines():
        if line.startswith("pid="):
            try:
                foreign_pid = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
            break

    kill_col, force_col = st.columns([1, 1])
    with kill_col:
        if st.button(
            f"Kill PID {foreign_pid}" if foreign_pid else "Kill running run",
            use_container_width=True,
            type="primary",
            disabled=foreign_pid is None,
        ):
            import os
            import signal

            try:
                os.kill(foreign_pid, signal.SIGTERM)
                # Give it a moment to release the lock cleanly
                for _ in range(20):
                    time.sleep(0.1)
                    try:
                        os.kill(foreign_pid, 0)  # still alive?
                    except (OSError, ProcessLookupError):
                        break
                else:
                    # Fallback: force-kill
                    try:
                        os.kill(foreign_pid, signal.SIGKILL)
                    except (AttributeError, OSError, ProcessLookupError):
                        pass
                st.toast(f"Stopped PID {foreign_pid}", icon=":material/stop_circle:")
            except (OSError, ProcessLookupError) as exc:
                st.toast(f"Could not stop PID {foreign_pid}: {exc}",
                         icon=":material/error:")
            # Clean up the (now-stale) lock file
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            st.rerun()
    with force_col:
        if st.button(
            "Delete stale lock", use_container_width=True,
            help="Use this if the lock looks stale but no process actually owns it.",
        ):
            try:
                lock_path.unlink(missing_ok=True)
                st.toast("Lock file removed", icon=":material/check:")
            except OSError as exc:
                st.toast(f"Could not delete lock: {exc}", icon=":material/error:")
            st.rerun()


def _build_cmd() -> List[str]:
    if st.session_state["an_source_kind"] == "youtube":
        source = st.session_state["an_url"]
    else:
        source = st.session_state["an_local_path"]
    cmd = [
        sys.executable, "-m", "futsal_analytics",
        "--url", source,
        "--start", st.session_state["an_start"] or "0",
        "--calibration", cal_path,
        "--no-gui",
        "--save-positions", str(out_dir / "positions.jsonl"),
        "--save-kpis", str(out_dir / "kpis.csv"),
        "--device", device,
        "--retrain-every", str(int(retrain_every)),
        "--log-level", "INFO",
        "--snapshot-every", str(int(snapshot_every)),
        "--snapshot-dir", str(out_dir),
        "--imgsz", str(int(imgsz)),
        "--frame-stride", str(int(frame_stride)),
        "--tracker-buffer", str(int(tracker_buffer)),
        "--tracker-conf", f"{float(tracker_conf):.2f}",
        "--tracker-iou", f"{float(tracker_iou):.2f}",
        "--tracker-minframes", str(int(tracker_minframes)),
    ]
    if rec_camera:
        cmd += ["--save-video", str(out_dir / "camera.mp4")]
    if rec_board:
        cmd += ["--save-board-video", str(out_dir / "board.mp4")]
    return cmd


def _start_reader(p: subprocess.Popen, q: queue.Queue) -> threading.Thread:
    def _pump() -> None:
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            if not line:
                break
            q.put(line)
        q.put(None)  # sentinel — process finished

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


with col_run:
    if st.button(
        "Start analysis",
        type="primary",
        use_container_width=True,
        disabled=running or not cal_ready or foreign_running,
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clear any stale live snapshots from a previous run so we don't show
        # the wrong match while the new one boots.
        for stale in ("_live_camera.jpg", "_live_board.jpg"):
            try:
                (out_dir / stale).unlink(missing_ok=True)
            except OSError:
                pass
        st.session_state["an_log_lines"] = []
        st.session_state["an_frames_done"] = 0
        st.session_state["an_started_at"] = time.time()
        st.session_state["an_finished_at"] = 0.0
        q: queue.Queue = queue.Queue()
        try:
            p = subprocess.Popen(
                _build_cmd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            st.error(f"Could not launch analyser: {exc}")
            p = None
        if p is not None:
            st.session_state["an_proc"] = p
            st.session_state["an_log_queue"] = q
            st.session_state["an_reader_thread"] = _start_reader(p, q)
            st.rerun()

with col_stop:
    if st.button(
        "Stop", use_container_width=True, disabled=not running,
    ):
        if proc is not None:
            try:
                proc.terminate()
            except Exception as exc:
                logging.warning("terminate failed: %s", exc)
        st.toast("Sent stop signal", icon=":material/stop_circle:")


# ---------------------------------------------------------------------------
# Live log + progress
# ---------------------------------------------------------------------------


def _render_snapshot(path: Path, caption: str) -> None:
    """Render a snapshot JPEG if present, reading it as bytes to bust cache."""
    if not path.exists():
        st.caption(f"_{caption} \u2014 waiting for first frame\u2026_")
        return
    try:
        data = path.read_bytes()
    except OSError:
        st.caption(f"_{caption} \u2014 (file busy, retrying)_")
        return
    st.image(data, caption=caption, use_container_width=True)


@st.fragment(run_every=0.3 if running else None)
def _live_log() -> None:
    proc_ = st.session_state.get("an_proc")
    q_ = st.session_state.get("an_log_queue")
    if proc_ is None or q_ is None:
        st.caption("Logs from the analyser will appear here when a run is in progress.")
        return

    # Drain newly-available lines from the queue
    finished_flag = False
    while True:
        try:
            line = q_.get_nowait()
        except queue.Empty:
            break
        if line is None:
            finished_flag = True
            break
        st.session_state["an_log_lines"].append(line.rstrip())
        m = PROGRESS_RE.search(line)
        if m:
            st.session_state["an_frames_done"] = int(m.group(1))

    if finished_flag or proc_.poll() is not None:
        if st.session_state["an_finished_at"] == 0.0:
            st.session_state["an_finished_at"] = time.time()

    started = st.session_state["an_started_at"]
    finished = st.session_state["an_finished_at"]
    elapsed = (finished or time.time()) - started if started else 0.0
    frames = st.session_state["an_frames_done"]

    cols = st.columns(4)
    cols[0].metric("Status", "Running" if proc_.poll() is None else "Finished")
    cols[1].metric("Frames", f"{frames:,}")
    cols[2].metric("Elapsed", f"{elapsed:.1f} s")
    cols[3].metric("FPS", f"{frames / elapsed:.1f}" if elapsed > 0 else "—")

    # Live previews (refreshes naturally because the fragment reruns every 1s)
    snap_cam = out_dir / "_live_camera.jpg"
    snap_board = out_dir / "_live_board.jpg"
    if snap_cam.exists() or snap_board.exists() or proc_.poll() is None:
        st.markdown("##### Live preview")
        prev_cols = st.columns(2, gap="medium")
        with prev_cols[0]:
            _render_snapshot(snap_cam, "Annotated camera")
        with prev_cols[1]:
            _render_snapshot(snap_board, "Tactical board")

    with st.expander("Log", expanded=False):
        log_text = "\n".join(st.session_state["an_log_lines"][-300:])
        st.code(log_text or "(no output yet)", language="text")

    if proc_.poll() is not None and proc_.poll() != 0 and st.session_state.get("_an_shown_error") != proc_.pid:
        st.error(f"Analyser exited with code {proc_.poll()}. See the log above.")
        st.session_state["_an_shown_error"] = proc_.pid
    elif proc_.poll() == 0 and st.session_state.get("_an_shown_done") != proc_.pid:
        st.success(
            "Analysis complete. Open the **Viewer** page in the sidebar to "
            "explore the results."
        )
        st.session_state["_an_shown_done"] = proc_.pid


st.markdown("#### Live log")
_live_log()
