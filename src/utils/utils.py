import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────

MP_HANDS = mp.solutions.hands

# Indices of the 5 fingertip keypoints (thumb, index, middle, ring, pinky)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandData:
    """
    All tracking data for a single hand in a single frame.

    Attributes
    ----------
    handedness : "Left" or "Right"
    keypoints  : (21, 2) float32 array of pixel coordinates
    bbox       : (x1, y1, x2, y2) bounding box in pixels
    """

    handedness: str
    keypoints:  np.ndarray   # shape (21, 2)
    bbox:       tuple        # (x1, y1, x2, y2)

    @property
    def wrist(self) -> np.ndarray:
        """Wrist keypoint (index 0), shape (2,)."""
        return self.keypoints[0]

    @property
    def centroid(self) -> np.ndarray:
        """Bounding-box center, shape (2,)."""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def fingertips(self) -> np.ndarray:
        """Coordinates of the 5 fingertips, shape (5, 2)."""
        return self.keypoints[FINGERTIP_INDICES]

    def hand_spread(self) -> float:
        """
        R(t): mean distance of each keypoint from the wrist (Eq. 6 of the paper).
        Measures how open the hand is.
        """
        dists = np.linalg.norm(self.keypoints[1:] - self.wrist, axis=1)
        return float(np.mean(dists))


@dataclass
class ObjectData:
    """
    Tracking data for a single object in a single frame.

    Attributes
    ----------
    label    : text label assigned by the VLM
    centroid : (cx, cy) pixel position, shape (2,)
    bbox     : (x1, y1, x2, y2) bounding box in pixels
    mask     : binary segmentation mask (H, W), dtype uint8, values 0/255
               None if the mask could not be computed
    """

    label:    str
    centroid: np.ndarray     # shape (2,)
    bbox:     tuple          # (x1, y1, x2, y2)
    mask:     np.ndarray     # shape (H, W), uint8, may be None


@dataclass
class FrameResult:
    """
    Aggregated tracking output for a single video frame.

    Attributes
    ----------
    frame_idx    : 0-based frame index
    timestamp_ms : timestamp in milliseconds
    hands        : list of HandData (one entry per detected hand)
    objects      : list of ObjectData (one entry per tracked object)
    """

    frame_idx:    int
    timestamp_ms: float
    hands:        list   # list[HandData]
    objects:      list   # list[ObjectData]

    def hand_object_distances(self) -> dict:
        """
        Compute pixel-space Euclidean distances between every hand centroid
        and every object centroid (Eq. 1 of the paper).

        Returns
        -------
        dict  { (hand_idx, object_label) -> float distance in pixels }
        """
        dists = {}
        for h_idx, hand in enumerate(self.hands):
            for obj in self.objects:
                d = float(np.linalg.norm(hand.centroid - obj.centroid))
                dists[(h_idx, obj.label)] = d
        return dists


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_first_frame(
    video_path: str,
) -> tuple[np.ndarray, float, int, int, int]:
    """
    Open a video file and return its first frame together with metadata.

    Parameters
    ----------
    video_path : path to the .mp4 file

    Returns
    -------
    frame   : np.ndarray  BGR frame (H, W, 3)
    fps     : float       frames per second
    total   : int         total number of frames
    frame_w : int         frame width  in pixels
    frame_h : int         frame height in pixels
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{video_path}'")

    fps     = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Could not read the first frame from the video.")

    print(f"[Video]  {frame_w} x {frame_h} px  |  {fps:.2f} fps  |  {total} frames")
    return frame, fps, total, frame_w, frame_h


# ─────────────────────────────────────────────────────────────────────────────
# VLM — object listing  (OpenAI API)
# ─────────────────────────────────────────────────────────────────────────────

_VLM_SYSTEM_PROMPT = (
    "You are an expert object detector specializing in table-top manipulation scenes."
)

_VLM_USER_PROMPT = (
    "Look at this image carefully. "
    "List all distinct objects that are placed on the table surface. "
    "Return ONLY a comma-separated list of short, unique object names. "
    "Do not include any explanation, preamble, or numbering. "
    "Example format:  red block, blue cup, yellow bottle, green bowl"
)


def _encode_frame_base64(frame: np.ndarray) -> str:
    """Convert a BGR OpenCV frame to a base64-encoded JPEG string."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def list_objects_vlm(
    frame:   np.ndarray,
    api_key: str,
    model:   str,
) -> list[str]:
    """
    Parameters
    ----------
    frame   : BGR first frame of the video
    api_key : API key  (use "API_KEY" as placeholder)
    model   : model name ("Molmo2")

    Returns
    -------
    labels : list of object label strings
    """
    client = OpenAI(api_key=api_key)
    b64    = _encode_frame_base64(frame)

    print(f"[VLM]  Querying {model} for object labels ...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            },
        ],
        max_tokens=200,
        temperature=0,
    )

    raw    = response.choices[0].message.content.strip()
    labels = [lbl.strip() for lbl in raw.split(",") if lbl.strip()]
    print(f"[VLM]  Detected {len(labels)} object(s): {labels}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# GroundingDINO — open-vocabulary object detection
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so the model is loaded only once per process
_DINO_PROCESSOR = None
_DINO_MODEL      = None
_DINO_DEVICE     = None


def detect_objects_dino(
    frame:     np.ndarray,
    labels:    list[str],
    threshold: float = 0.30,
) -> dict[str, tuple]:
    """
    Run GroundingDINO on a single frame given a list of text labels.

    Parameters
    ----------
    frame     : BGR frame (numpy array)
    labels    : list of object label strings (from list_objects_vlm)
    threshold : confidence threshold for both box and text scores

    Returns
    -------
    detections : dict  { label -> (cx, cy, x1, y1, x2, y2, score) }
                 One entry per label; the highest-confidence box is kept.
    """
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )

    global _DINO_PROCESSOR, _DINO_MODEL, _DINO_DEVICE

    # ── Lazy-load model ───────────────────────────────────────────────────────
    if _DINO_PROCESSOR is None:
        _DINO_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
        model_id       = "IDEA-Research/grounding-dino-base"
        print(f"[DINO]  Loading '{model_id}' on {_DINO_DEVICE} ...")
        _DINO_PROCESSOR = AutoProcessor.from_pretrained(model_id)
        _DINO_MODEL     = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(_DINO_DEVICE)
        print("[DINO]  Model ready.")

    # ── Build text query ──────────────────────────────────────────────────────
    # GroundingDINO expects labels joined with " . " and ending with " ."
    text_query = " . ".join(labels) + " ."

    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = _DINO_PROCESSOR(
        images=pil_img,
        text=text_query,
        return_tensors="pt",
    ).to(_DINO_DEVICE)

    with torch.no_grad():
        outputs = _DINO_MODEL(**inputs)

    results = _DINO_PROCESSOR.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(h, w)],
    )[0]

    # ── Build output dict ─────────────────────────────────────────────────────
    detections: dict[str, tuple] = {}
    boxes            = results["boxes"].cpu().numpy()   # (N, 4) xyxy
    scores           = results["scores"].cpu().numpy()  # (N,)
    detected_labels  = results["labels"]                # list[str]

    for box, score, det_label in zip(boxes, scores, detected_labels):
        x1, y1, x2, y2 = box.astype(int)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        matched = _match_label(det_label, labels)
        if matched is None:
            continue  # skip detections that do not match any VLM label

        # Keep the highest-score detection for each label
        if matched not in detections or score > detections[matched][-1]:
            detections[matched] = (cx, cy, x1, y1, x2, y2, float(score))

    summary = {k: f"score={v[-1]:.2f}" for k, v in detections.items()}
    print(f"[DINO]  Detections: {summary}")
    return detections


def _match_label(detected: str, candidates: list[str]) -> str | None:
    """
    Match a GroundingDINO output phrase to the closest user-provided label.

    Tries exact match first, then substring match in both directions.
    Returns None if no candidate matches.
    """
    d = detected.lower().strip()
    for c in candidates:
        if c.lower().strip() == d:
            return c
    for c in candidates:
        cl = c.lower().strip()
        if cl in d or d in cl:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation mask
# ─────────────────────────────────────────────────────────────────────────────

def _make_bbox_mask(frame: np.ndarray, bbox_xyxy: tuple) -> np.ndarray:
    """
    Generate a binary segmentation mask for an object given its bounding box.

    Parameters
    ----------
    frame     : BGR frame
    bbox_xyxy : (x1, y1, x2, y2) integer bounding box

    Returns
    -------
    mask : uint8 array (H, W), values 0 or 255
    """
    h, w    = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    # Clamp to frame boundaries
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(w, x2);  y2 = min(h, y2)
    bw = x2 - x1
    bh = y2 - y1

    # GrabCut requires a minimum region size to be meaningful
    if bw < 10 or bh < 10:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    try:
        gc_mask   = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            frame, gc_mask, (x1, y1, bw, bh),
            bgd_model, fgd_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_RECT,
        )
        binary = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            np.uint8(255),
            np.uint8(0),
        )
        return binary

    except cv2.error:
        # Fallback: simple rectangular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Multi-object tracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiObjectTracker:
    """
    Tracks multiple labeled objects across video frames.
    Parameters
    ----------
    frame          : BGR first frame of the video
    detections     : output of detect_objects_dino() on the first frame
    redetect_every : re-run DINO every N frames  (0 = never re-detect)
    dino_threshold : confidence threshold used during re-detection
    """

    def __init__(
        self,
        frame:          np.ndarray,
        detections:     dict[str, tuple],
        redetect_every: int   = 30,
        dino_threshold: float = 0.30,
    ):
        self.labels         = list(detections.keys())
        self.redetect_every = redetect_every
        self.dino_threshold = dino_threshold
        self._frame_count   = 0

        # Internal state: one tracker + last known bbox per object
        self._trackers: dict[str, cv2.TrackerCSRT] = {}
        self._bboxes:   dict[str, tuple]           = {}   # (x, y, w, h) OpenCV format

        for label, det in detections.items():
            _cx, _cy, x1, y1, x2, y2, _score = det
            self._init_tracker(frame, label, x1, y1, x2, y2)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_tracker(
        self,
        frame: np.ndarray,
        label: str,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Create and initialize a CSRT tracker for *label* from a xyxy box."""
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, bw, bh))
        self._trackers[label] = tracker
        self._bboxes[label]   = (x1, y1, bw, bh)

    def _redetect(self, frame: np.ndarray) -> None:
        """Re-run GroundingDINO and reinitialize trackers for re-detected objects."""
        try:
            new_dets = detect_objects_dino(frame, self.labels, self.dino_threshold)
            for label, det in new_dets.items():
                _cx, _cy, x1, y1, x2, y2, _score = det
                self._init_tracker(frame, label, x1, y1, x2, y2)
        except Exception as exc:
            print(f"[DINO]  Re-detection failed (frame {self._frame_count}): {exc}")

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> dict[str, ObjectData]:
        """
        Advance all trackers by one frame and return per-object tracking data.

        Parameters
        ----------
        frame : BGR frame

        Returns
        -------
        dict  { label -> ObjectData }
        """
        self._frame_count += 1
        fh, fw = frame.shape[:2]

        # Periodic re-detection to prevent tracker drift
        if self.redetect_every > 0 and self._frame_count % self.redetect_every == 0:
            self._redetect(frame)

        objects: dict[str, ObjectData] = {}

        for label, tracker in self._trackers.items():
            ok, bbox_xywh = tracker.update(frame)

            if ok:
                x, y, bw, bh = [int(v) for v in bbox_xywh]
                # Clamp to frame boundaries to avoid out-of-bounds accesses
                x  = max(0, min(x,  fw - 1))
                y  = max(0, min(y,  fh - 1))
                bw = max(1, min(bw, fw - x))
                bh = max(1, min(bh, fh - y))
                self._bboxes[label] = (x, y, bw, bh)
            else:
                # Tracker lost; fall back to last known bounding box
                x, y, bw, bh = self._bboxes.get(label, (0, 0, 1, 1))

            x1, y1, x2, y2 = x, y, x + bw, y + bh
            cx, cy          = x + bw // 2, y + bh // 2
            mask            = _make_bbox_mask(frame, (x1, y1, x2, y2))

            objects[label] = ObjectData(
                label=label,
                centroid=np.array([cx, cy], dtype=np.float32),
                bbox=(x1, y1, x2, y2),
                mask=mask,
            )

        return objects


# ─────────────────────────────────────────────────────────────────────────────
# Hand tracker  (MediaPipe Hands)
# ─────────────────────────────────────────────────────────────────────────────

class HandTracker:
    """
    Wraps MediaPipe Hands for per-frame hand keypoint detection.

    Parameters
    ----------
    max_num_hands            : maximum number of hands to detect
    min_detection_confidence : MediaPipe detection confidence threshold
    min_tracking_confidence  : MediaPipe tracking confidence threshold
    """

    def __init__(
        self,
        max_num_hands:            int   = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ):
        self._hands = MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[HandData]:
        """
        Detect hands in a single BGR frame.

        Returns
        -------
        list[HandData]  — one entry per detected hand, ordered as returned
                          by MediaPipe.
        """
        h, w      = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self._hands.process(frame_rgb)
        hands     = []

        if not results.multi_hand_landmarks:
            return hands

        for lm_list, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            handedness = handedness_info.classification[0].label  # "Left" / "Right"

            # Convert normalized landmarks to pixel coordinates
            keypoints = np.array(
                [[lm.x * w, lm.y * h] for lm in lm_list.landmark],
                dtype=np.float32,
            )  # shape (21, 2)

            # Build bounding box from the keypoint cloud + padding
            xs, ys = keypoints[:, 0], keypoints[:, 1]
            pad    = 15
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(w, int(xs.max()) + pad)
            y2 = min(h, int(ys.max()) + pad)

            hands.append(HandData(
                handedness=handedness,
                keypoints=keypoints,
                bbox=(x1, y1, x2, y2),
            ))

        return hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


# ─────────────────────────────────────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────────────────────────────────────

def compute_relative_distance(
    hand_traj: list,   # list[np.ndarray | None]
    obj_traj:  list,   # list[np.ndarray | None]
) -> list:
    """
    Compute the hand-object distance d_i(t) for every frame

        d_i(t) = || p_h(t) - p_oi(t) ||

    Parameters
    ----------
    hand_traj : list of (2,) hand centroid arrays, or None when not detected
    obj_traj  : list of (2,) object centroid arrays, or None when not detected

    Returns
    -------
    distances : list of float  (None where either position is missing)
    """
    distances = []
    for ph, po in zip(hand_traj, obj_traj):
        if ph is None or po is None:
            distances.append(None)
        else:
            distances.append(float(np.linalg.norm(ph - po)))
    return distances


def compute_relative_velocity(
    distances:     list,   # list[float | None]
    timestamps_ms: list,   # list[float]
) -> list:
    """
    Compute relative hand-object velocity v_oi(t) = d(d_i)/dt by finite differences.

    A positive value means the hand is moving away from the object;
    a negative value means the hand is approaching.

    Parameters
    ----------
    distances     : output of compute_relative_distance()
    timestamps_ms : frame timestamps in milliseconds

    Returns
    -------
    rel_velocities : list of float  (None where undefined)
                     First element is always None.
    """
    rel_velocities: list = [None]
    for t in range(1, len(distances)):
        if distances[t] is None or distances[t - 1] is None:
            rel_velocities.append(None)
        else:
            dt = (timestamps_ms[t] - timestamps_ms[t - 1]) / 1000.0
            if dt <= 0:
                dt = 1.0 / 30.0  # fallback: assume 30 fps
            rel_velocities.append((distances[t] - distances[t - 1]) / dt)
    return rel_velocities


# ─────────────────────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results_txt(
    frame_results: list,   # list[FrameResult]
    output_path:   str,
    fps:           float,
) -> None:
    """
    Write all per-frame tracking results to a formatted plain-text file.

    File format
    ----------------------------
    Header block with video metadata.

    Per-frame block:
      FRAME <idx>  t=<ms> ms
        HAND  <side>  centroid=(<x>, <y>)  wrist=(<x>, <y>)
              spread=<R> px  bbox=(<x1>,<y1>,<x2>,<y2>)
        OBJECT <label>  centroid=(<x>, <y>)  bbox=(...)  mask=<yes|no>
        DISTANCE  hand[<idx>:<side>] -> <label> : <d> px

    Parameters
    ----------
    frame_results : list[FrameResult]
    output_path   : destination .txt file path (parent dirs created if needed)
    fps           : video frame rate (for header info only)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sep_heavy = "=" * 70
    sep_light = "-" * 70

    with open(output_path, "w", encoding="utf-8") as f:

        # ── File header ───────────────────────────────────────────────────────
        f.write(sep_heavy + "\n")
        f.write("  HAND-OBJECT TRACKING RESULTS\n")
        f.write(sep_heavy + "\n")
        f.write(f"  Video FPS      : {fps:.2f}\n")
        f.write(f"  Frames stored  : {len(frame_results)}\n")

        # Collect unique object labels from all frames
        all_labels = sorted({
            obj.label
            for fr in frame_results
            for obj in fr.objects
        })
        f.write(f"  Tracked objects: {', '.join(all_labels) if all_labels else 'none'}\n")
        f.write(sep_heavy + "\n\n")

        # ── Per-frame blocks ──────────────────────────────────────────────────
        for fr in frame_results:
            f.write(f"FRAME {fr.frame_idx:05d}  t={fr.timestamp_ms:.1f} ms\n")
            f.write(sep_light + "\n")

            # Hand entries
            if fr.hands:
                for hand in fr.hands:
                    cx, cy = hand.centroid
                    wx, wy = hand.wrist
                    x1, y1, x2, y2 = hand.bbox
                    f.write(
                        f"  HAND  {hand.handedness:<6s}  "
                        f"centroid=({cx:7.1f}, {cy:7.1f})  "
                        f"wrist=({wx:7.1f}, {wy:7.1f})  "
                        f"spread={hand.hand_spread():6.2f} px  "
                        f"bbox=({x1},{y1},{x2},{y2})\n"
                    )
            else:
                f.write("  HAND  [not detected]\n")

            # Object entries
            if fr.objects:
                for obj in fr.objects:
                    cx, cy     = obj.centroid
                    x1, y1, x2, y2 = obj.bbox
                    has_mask   = "yes" if obj.mask is not None else "no"
                    f.write(
                        f"  OBJECT  {obj.label:<30s}  "
                        f"centroid=({cx:7.1f}, {cy:7.1f})  "
                        f"bbox=({x1},{y1},{x2},{y2})  "
                        f"mask={has_mask}\n"
                    )
            else:
                f.write("  OBJECT  [not detected]\n")

            # Hand-object distances
            dists = fr.hand_object_distances()
            for (h_idx, label), d in dists.items():
                side = fr.hands[h_idx].handedness if fr.hands else "?"
                f.write(
                    f"  DISTANCE  hand[{h_idx}:{side}] -> "
                    f"{label:<30s}: {d:8.1f} px\n"
                )

            f.write("\n")

    print(f"[Output]  Results saved  →  {output_path}")