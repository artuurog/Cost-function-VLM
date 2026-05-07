"""
=================
Full hand-object tracking pipeline for an RGB video.

Dependencies
------------
  pip install opencv-python mediapipe openai pillow numpy
  pip install torch torchvision transformers   # for GroundingDINO
"""

from pathlib import Path

import cv2
import numpy as np

from utils import (
    # Data structures
    FrameResult,
    # Video
    extract_first_frame,
    # VLM
    list_objects_vlm,
    # Detection & tracking
    detect_objects_dino,
    MultiObjectTracker,
    HandTracker,
    # Kinematics
    compute_relative_distance,
    compute_relative_velocity,
    # I/O
    save_results_txt,
)


# ─────────────────────────────────────────────────────────────────────────────
# User settings  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_PATH  = "C:/Users/user/Desktop/PoliMi/DOTTORATO/hand object interaction/video/my_demos/red_block1.mp4"
OUTPUT_PATH = "results/tracking_results.txt"

# OpenAI settings
OPENAI_API_KEY = "API_KEY"     # replace with your actual key
VLM_MODEL      = "Molmo2"     # model used for object listing

# GroundingDINO settings
DINO_THRESHOLD  = 0.30         # detection confidence threshold
REDETECT_EVERY  = 30           # re-run DINO every N frames (0 = disabled)

# MediaPipe hand tracking settings
MAX_HANDS               = 2
HAND_DETECT_CONFIDENCE  = 0.5
HAND_TRACK_CONFIDENCE   = 0.5

# Processing settings
DISPLAY      = True     # show annotated video in real time (press Q to quit)
SKIP_FRAMES  = 0        # set > 0 to subsample (e.g. 1 = analyze every other frame)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_frame(
    frame:   np.ndarray,
    hands:   list,   # list[HandData]
    objects: dict,   # dict[str, ObjectData]
) -> np.ndarray:
    """
    Draw hand keypoints/bboxes and object bboxes/centroids on a copy of the frame.

    Color coding:
      Right hand  →  green  (0, 255, 127)
      Left  hand  →  orange (255, 165, 0)
      Objects     →  cyan   (0, 200, 255)
    """
    out = frame.copy()

    HAND_COLORS = {"Right": (0, 255, 127), "Left": (255, 165, 0)}
    OBJ_COLOR   = (0, 200, 255)

    # ── Hands ─────────────────────────────────────────────────────────────────
    for hand in hands:
        color = HAND_COLORS.get(hand.handedness, (200, 200, 200))

        # Skeleton connections defined by MediaPipe
        for conn in cv2.solutions.hands.HAND_CONNECTIONS if hasattr(cv2, 'solutions') else []:
            pt1 = hand.keypoints[conn[0]].astype(int)
            pt2 = hand.keypoints[conn[1]].astype(int)
            cv2.line(out, tuple(pt1), tuple(pt2), color, 1, cv2.LINE_AA)

        # Keypoints
        for kp in hand.keypoints:
            cv2.circle(out, (int(kp[0]), int(kp[1])), 3, color, -1)

        # Bounding box and label
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out, hand.handedness, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        # Centroid cross
        c = hand.centroid.astype(int)
        cv2.drawMarker(out, tuple(c), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # ── Objects ───────────────────────────────────────────────────────────────
    for label, obj in objects.items():
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), OBJ_COLOR, 2)
        c = obj.centroid.astype(int)
        cv2.circle(out, tuple(c), 5, OBJ_COLOR, -1)
        cv2.putText(
            out, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, OBJ_COLOR, 1, cv2.LINE_AA,
        )

        # Overlay segmentation mask with transparency when available
        if obj.mask is not None:
            overlay        = out.copy()
            colored_mask   = np.zeros_like(frame)
            colored_mask[:, :] = OBJ_COLOR
            overlay[obj.mask == 255] = (
                overlay[obj.mask == 255] * 0.6 +
                colored_mask[obj.mask == 255] * 0.4
            ).astype(np.uint8)
            out = overlay

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    video_path:  str,
    output_path: str,
    api_key:     str,
) -> list:
    """
    Run the complete hand-object tracking pipeline on a video file.

    Parameters
    ----------
    video_path  : path to the input .mp4 video
    output_path : path for the output .txt results file
    api_key     : OpenAI API key for the VLM object-listing step

    Returns
    -------
    frame_results : list[FrameResult]
        One FrameResult per processed frame, containing hand and object
        positions, bounding boxes, and segmentation masks.
    """

    # ─── STEP 1 — Extract first frame ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Extract first frame")
    print("=" * 60)
    first_frame, fps, total_frames, frame_w, frame_h = extract_first_frame(video_path)

    # ─── STEP 2 — VLM object listing ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — VLM object listing  (OpenAI)")
    print("=" * 60)
    object_labels = list_objects_vlm(first_frame, api_key, model=VLM_MODEL)

    if not object_labels:
        raise RuntimeError(
            "The VLM returned no object labels. "
            "Verify the API key and that the first frame is informative."
        )

    # ─── STEP 3 — GroundingDINO detection on first frame ─────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — GroundingDINO detection on first frame")
    print("=" * 60)
    detections = detect_objects_dino(first_frame, object_labels, DINO_THRESHOLD)

    if not detections:
        raise RuntimeError(
            "GroundingDINO found no objects above the confidence threshold. "
            f"Try lowering DINO_THRESHOLD (currently {DINO_THRESHOLD})."
        )

    # ─── STEP 4 — Initialize object and hand trackers ────────────────────────
    obj_tracker  = MultiObjectTracker(
        first_frame, detections,
        redetect_every=REDETECT_EVERY,
        dino_threshold=DINO_THRESHOLD,
    )
    hand_tracker = HandTracker(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=HAND_DETECT_CONFIDENCE,
        min_tracking_confidence=HAND_TRACK_CONFIDENCE,
    )

    # ─── STEP 5 — Frame-by-frame processing ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Processing video frame by frame")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for processing: '{video_path}'")

    all_results: list[FrameResult] = []
    frame_idx   = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # end of video

            # Optional frame subsampling
            if SKIP_FRAMES > 0 and frame_idx % (SKIP_FRAMES + 1) != 0:
                frame_idx += 1
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Hand detection (MediaPipe)
            hands = hand_tracker.detect(frame_bgr)

            # Object tracking (CSRT + optional DINO re-detection)
            # Returns dict[str, ObjectData]; convert to list for FrameResult
            objects_dict = obj_tracker.update(frame_bgr)
            objects_list = list(objects_dict.values())

            # Store aggregated frame result
            fr = FrameResult(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                hands=hands,
                objects=objects_list,
            )
            all_results.append(fr)

            # Progress log every 30 analysed frames
            if frame_idx % 30 == 0:
                print(
                    f"  [Frame {frame_idx:5d} / {total_frames}]  "
                    f"hands={len(hands)}  objects={len(objects_list)}"
                )

            # Optional live display
            if DISPLAY:
                annotated = _annotate_frame(frame_bgr, hands, objects_dict)
                cv2.imshow("Track Combined  (press Q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO]  User interrupted (Q key).")
                    break

            frame_idx += 1

    finally:
        cap.release()
        hand_tracker.close()
        if DISPLAY:
            cv2.destroyAllWindows()

    print(f"\n[INFO]  Processed {len(all_results)} frames.")

    # ─── STEP 6 — Save results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 — Saving results")
    print("=" * 60)
    save_results_txt(all_results, output_path, fps)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectories(
    frame_results: list,
    hand_idx:      int = 0,
    obj_label:     str = "",
) -> tuple[list, list, list]:
    """
    Extract position trajectories and timestamps from the pipeline output.

    Parameters
    ----------
    frame_results : list[FrameResult]  output of run_pipeline()
    hand_idx      : which hand to extract (0 = first detected hand)
    obj_label     : label of the object to extract

    Returns
    -------
    hand_traj     : list of (2,) centroid arrays or None per frame
    obj_traj      : list of (2,) centroid arrays or None per frame
    timestamps_ms : list of float
    """
    hand_traj:     list = []
    obj_traj:      list = []
    timestamps_ms: list = []

    for fr in frame_results:
        timestamps_ms.append(fr.timestamp_ms)

        # Hand position
        if len(fr.hands) > hand_idx:
            hand_traj.append(fr.hands[hand_idx].centroid.copy())
        else:
            hand_traj.append(None)

        # Object position
        obj_data = next((o for o in fr.objects if o.label == obj_label), None)
        obj_traj.append(obj_data.centroid.copy() if obj_data is not None else None)

    return hand_traj, obj_traj, timestamps_ms


def print_kinematic_summary(
    frame_results: list,
    fps:           float,
) -> None:
    """
    Print a kinematic summary (mean distance and mean relative speed) for
    every hand-object pair detected in the first frame.
    """
    if not frame_results:
        return

    # Collect object labels from the first frame that has objects
    obj_labels = []
    for fr in frame_results:
        if fr.objects:
            obj_labels = [o.label for o in fr.objects]
            break

    if not obj_labels:
        print("[Summary]  No objects detected in any frame.")
        return

    print("\n" + "=" * 60)
    print("  KINEMATIC SUMMARY  (hand[0] relative to each object)")
    print("=" * 60)

    for label in obj_labels:
        hand_traj, obj_traj, ts = extract_trajectories(
            frame_results, hand_idx=0, obj_label=label
        )
        rel_dist = compute_relative_distance(hand_traj, obj_traj)
        rel_vel  = compute_relative_velocity(rel_dist, ts)

        valid_d = [d for d in rel_dist if d is not None]
        valid_v = [v for v in rel_vel  if v is not None]

        if valid_d:
            print(
                f"  {label:<30s}  "
                f"mean dist = {np.mean(valid_d):7.1f} px  |  "
                f"mean |v_rel| = {np.mean(np.abs(valid_v)):7.1f} px/s"
            )
        else:
            print(f"  {label:<30s}  [no valid data]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if not Path(VIDEO_PATH).is_file():
        raise FileNotFoundError(f"Video not found: '{VIDEO_PATH}'")

    # Run the full pipeline
    results = run_pipeline(VIDEO_PATH, OUTPUT_PATH, OPENAI_API_KEY)

    # Print kinematic summary to console
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print_kinematic_summary(results, fps)

    print("\n[DONE]")