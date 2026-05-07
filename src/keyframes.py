"""
extract_keyframes.py
====================

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# User settings — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH = (
    "C:/Users/user/Desktop/PoliMi/DOTTORATO/"
    "hand object interaction/video/my_demos/red_block1.mp4"
)

# Path to the interaction_probability.txt file
PROBABILITY_FILE = "results/interaction_probability.txt"

# Output directory where extracted keyframes will be saved
OUTPUT_DIR = "results/keyframes"

# "files" → one JPEG per keyframe  |  "video" → single summary .mp4 clip
EXTRACTION_MODE: str = "files"

# JPEG quality for image output (1–100)
JPEG_QUALITY: int = 95

# When saving as video: frames-per-second of the output clip
SUMMARY_VIDEO_FPS: float = 2.0

# If True and PROBABILITY_FILE is missing, run the full pipeline automatically.
# Requires interaction_probability.py to be importable from the same directory.
RUN_PIPELINE_IF_NEEDED: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Parser for interaction_probability.txt
# ─────────────────────────────────────────────────────────────────────────────

def parse_keyframe_table(
    path: str,
) -> List[Tuple[int, float, str, float]]:
    """
    Returns
    -------
    keyframes : list of (frame_idx, timestamp_ms, dominant_object, probability)
                sorted by frame_idx (chronological order).
    """
    keyframes: List[Tuple[int, float, str, float]] = []
    in_table = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()

            if "KEYFRAME TABLE" in line:
                in_table = True
                continue

            # Skip comment / header lines
            if line.startswith("#"):
                continue

            if not in_table or not line:
                continue

            # Data row: rank  frame_idx  timestamp_ms  dominant_object  probability
            tokens = line.split()
            if len(tokens) < 5:
                continue
            try:
                # rank        = int(tokens[0])   # not needed
                frame_idx   = int(tokens[1])
                timestamp   = float(tokens[2])
                dom_object  = tokens[3]
                probability = float(tokens[4])
            except ValueError:
                continue

            keyframes.append((frame_idx, timestamp, dom_object, probability))

    # Sort chronologically (should already be sorted, but make it explicit)
    keyframes.sort(key=lambda x: x[0])
    return keyframes


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: '{video_path}'")
    return cap


def _annotate(
    frame:       np.ndarray,
    frame_idx:   int,
    timestamp:   float,
    dom_object:  str,
    probability: float,
    rank:        int,
) -> np.ndarray:

    
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent dark banner at the bottom
    banner_h = 60
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    color      = (255, 255, 255)
    y0         = h - banner_h + 18

    line1 = f"Keyframe #{rank}  |  frame {frame_idx}  |  t = {timestamp:.1f} ms"
    line2 = f"Dominant object: {dom_object}   P = {probability:.4f}"

    cv2.putText(out, line1, (10, y0),          font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(out, line2, (10, y0 + 22),     font, font_scale, (0, 220, 255), thickness, cv2.LINE_AA)

    return out


def extract_keyframes_to_files(
    video_path:  str,
    keyframes:   List[Tuple[int, float, str, float]],
    output_dir:  str,
    jpeg_quality: int = 95,
) -> List[str]:
    """
    Save each keyframe as a JPEG file inside output_dir.

    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap   = _open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_paths: List[str] = []

    for rank, (frame_idx, timestamp, dom_object, probability) in enumerate(keyframes, start=1):

        if frame_idx >= total:
            print(f"  [WARN] frame_idx={frame_idx} exceeds video length ({total}); skipping.")
            continue

        # Seek directly to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"  [WARN] Could not read frame {frame_idx}; skipping.")
            continue

        # Annotate the frame with metadata
        annotated = _annotate(frame, frame_idx, timestamp, dom_object, probability, rank)

        # Build output filename (replace spaces in object name with underscores)
        safe_label = dom_object.replace(" ", "_")
        filename   = f"keyframe_{rank:03d}_frame{frame_idx:05d}_{safe_label}.jpg"
        filepath   = str(out_dir / filename)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        cv2.imwrite(filepath, annotated, encode_params)
        saved_paths.append(filepath)

        print(
            f"  [{rank:3d}/{len(keyframes)}]  frame {frame_idx:5d}  "
            f"t={timestamp:8.1f} ms  P={probability:.4f}  "
            f"obj={dom_object:<20s}  →  {filename}"
        )

    cap.release()
    return saved_paths


def extract_keyframes_to_video(
    video_path:    str,
    keyframes:     List[Tuple[int, float, str, float]],
    output_dir:    str,
    summary_fps:   float = 2.0,
) -> str:
    """
    Assemble all keyframes into a single summary .mp4 clip.

    Returns
    -------
    output_path : path to the written .mp4 file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap        = _open_video(video_path)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_stem   = Path(video_path).stem
    output_path  = str(out_dir / f"{video_stem}_keyframe_summary.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, summary_fps, (frame_w, frame_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for '{output_path}'")

    for rank, (frame_idx, timestamp, dom_object, probability) in enumerate(keyframes, start=1):

        if frame_idx >= total:
            print(f"  [WARN] frame_idx={frame_idx} exceeds video length ({total}); skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"  [WARN] Could not read frame {frame_idx}; skipping.")
            continue

        annotated = _annotate(frame, frame_idx, timestamp, dom_object, probability, rank)
        writer.write(annotated)

        print(
            f"  [{rank:3d}/{len(keyframes)}]  frame {frame_idx:5d}  "
            f"t={timestamp:8.1f} ms  P={probability:.4f}  obj={dom_object}"
        )

    cap.release()
    writer.release()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    prob_path = Path(PROBABILITY_FILE)

    if not prob_path.is_file():
        if RUN_PIPELINE_IF_NEEDED:
            print(
                f"[INFO] '{PROBABILITY_FILE}' not found. "
                "Running interaction_probability pipeline..."
            )
            try:
                from interaction_probability import compute_interaction_probability
                compute_interaction_probability(
                    cost_path   = "results/cost_function.txt",
                    output_path = PROBABILITY_FILE,
                )
            except ImportError:
                print(
                    "[ERROR] Cannot import interaction_probability.py. "
                    "Make sure it is in the same directory and "
                    "its dependencies (scipy, numpy) are installed."
                )
                sys.exit(1)
        else:
            print(
                f"[ERROR] Probability file not found: '{PROBABILITY_FILE}'\n"
                "        Run interaction_probability.py first, or set "
                "RUN_PIPELINE_IF_NEEDED = True."
            )
            sys.exit(1)

    # ── Check video file ──────────────────────────────────────────────────────
    if not Path(VIDEO_PATH).is_file():
        print(f"[ERROR] Video not found: '{VIDEO_PATH}'")
        sys.exit(1)

    # ── Parse keyframe table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Parse keyframe table")
    print("=" * 60)

    keyframes = parse_keyframe_table(PROBABILITY_FILE)

    if not keyframes:
        print("[ERROR] No keyframes found in the probability file.")
        sys.exit(1)

    print(f"  Found {len(keyframes)} keyframe(s):")
    for rank, (fi, ts, obj, prob) in enumerate(keyframes, start=1):
        print(f"    #{rank:3d}  frame={fi:5d}  t={ts:8.1f} ms  "
              f"P={prob:.4f}  obj={obj}")

    # ── Extract frames ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  STEP 2 — Extract keyframes  (mode: '{EXTRACTION_MODE}')")
    print("=" * 60 + "\n")

    if EXTRACTION_MODE == "files":
        saved = extract_keyframes_to_files(
            video_path   = VIDEO_PATH,
            keyframes    = keyframes,
            output_dir   = OUTPUT_DIR,
            jpeg_quality = JPEG_QUALITY,
        )
        print(f"\n[DONE]  {len(saved)} keyframe image(s) saved to '{OUTPUT_DIR}/'")

    elif EXTRACTION_MODE == "video":
        out_path = extract_keyframes_to_video(
            video_path  = VIDEO_PATH,
            keyframes   = keyframes,
            output_dir  = OUTPUT_DIR,
            summary_fps = SUMMARY_VIDEO_FPS,
        )
        print(f"\n[DONE]  Summary video saved to '{out_path}'")

    else:
        print(
            f"[ERROR] Unknown EXTRACTION_MODE '{EXTRACTION_MODE}'. "
            "Choose 'files' or 'video'."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()