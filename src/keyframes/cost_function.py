
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


TRACKING_RESULTS_PATH = "results/tracking_results.txt"
COST_OUTPUT_PATH      = "results/cost_function.txt"

# Fingertip keypoint indices 
FINGERTIP_INDICES: List[int] = [4, 8, 12, 16, 20]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandSnapshot:
    """Per-frame hand state extracted from the tracking file."""
    handedness: str                          # "Left" | "Right"
    centroid:   np.ndarray                   # shape (2,)  [cx, cy]
    keypoints:  np.ndarray                   # shape (21, 2)
    bbox:       Tuple[int, int, int, int]    # (x1, y1, x2, y2)

    @property
    def wrist(self) -> np.ndarray:
        return self.keypoints[0]

    @property
    def fingertips(self) -> np.ndarray:
        """Shape (5, 2)."""
        return self.keypoints[FINGERTIP_INDICES]

    def hand_spread(self) -> float:
        """R(t): mean distance of each keypoint from the wrist."""
        dists = np.linalg.norm(self.keypoints[1:] - self.wrist, axis=1)
        return float(np.mean(dists))


@dataclass
class ObjectSnapshot:
    """Per-frame object state extracted from the tracking file."""
    label:    str
    centroid: np.ndarray                   # shape (2,)  [cx, cy]
    bbox:     Tuple[int, int, int, int]    # (x1, y1, x2, y2)


@dataclass
class FrameRecord:
    """
    All tracking data for a single video frame.

    """
    frame_idx:    int
    timestamp_ms: float
    hand:         Optional[HandSnapshot]           # primary hand
    hand2:        Optional[HandSnapshot] = None    # secondary hand (two-hand mode)
    objects:      Dict[str, ObjectSnapshot] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking results parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_tracking_file(path: str) -> Tuple[List[FrameRecord], float, int]:
    """
    Read a tracking-results .txt file and return:
      - list of FrameRecord (one per frame)
      - video FPS (parsed from header comment, default 30.0)
      - num_hands (parsed from header comment, default 1)

    """
    records: List[FrameRecord] = []
    fps       = 30.0
    num_hands = 1
    current: Optional[FrameRecord] = None

    # Regex helpers for the human-readable format produced by utils.py
    _re_centroid = re.compile(r"centroid=\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)")
    _re_bbox     = re.compile(r"bbox=\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")

    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # ── Header comments ───────────────────────────────────────────
            if line.startswith("#"):
                # FPS
                m = re.search(r"FPS[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if m:
                    fps = float(m.group(1))
                # num_hands
                m2 = re.search(r"num_hands[:\s=]+([0-9]+)", line, re.IGNORECASE)
                if m2:
                    num_hands = int(m2.group(1))
                continue

            # Header lines that start with spaces (utils.py format) may carry
            # num_hands / FPS information outside comments — capture them too.
            m_fps = re.search(r"Video FPS\s*:\s*([0-9.]+)", line, re.IGNORECASE)
            if m_fps:
                fps = float(m_fps.group(1))
            m_nh = re.search(r"num_hands\s*[=:]\s*([0-9]+)", line, re.IGNORECASE)
            if m_nh:
                num_hands = int(m_nh.group(1))

            tokens = line.split()
            if not tokens:
                continue
            tag = tokens[0].upper()

            # ── FRAME line ────────────────────────────────────────────────
            if tag == "FRAME":
                if current is not None:
                    records.append(current)
                frame_idx    = int(tokens[1])
                timestamp_ms = 0.0
                for tok in tokens[2:]:
                    m_ts = re.match(r"t=([0-9.]+)", tok)
                    if m_ts:
                        timestamp_ms = float(m_ts.group(1))
                        break
                current = FrameRecord(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    hand=None,
                    hand2=None,
                )

            # ── HAND line ─────────────────────────────────────────────────
            elif tag == "HAND" and current is not None:
                hand_snap = _parse_hand_line(
                    tokens, line, _re_centroid, _re_bbox
                )
                if hand_snap is None:
                    continue
                # Assign to primary or secondary slot
                if current.hand is None:
                    current.hand = hand_snap
                elif current.hand2 is None:
                    current.hand2 = hand_snap

            # ── OBJECT line ───────────────────────────────────────────────
            elif tag == "OBJECT" and current is not None:
                obj_snap = _parse_object_line(tokens)
                if obj_snap is not None:
                    current.objects[obj_snap.label] = obj_snap

    if current is not None:
        records.append(current)

    print(
        f"[Parser] Loaded {len(records)} frames from '{path}'  "
        f"(fps={fps:.2f}, num_hands={num_hands})"
    )
    return records, fps, num_hands


def _parse_hand_line(
    tokens:       List[str],
    raw_line:     str,
    re_centroid:  re.Pattern,
    re_bbox:      re.Pattern,
) -> Optional[HandSnapshot]:

    if len(tokens) < 2:
        return None

    handedness = tokens[1]   # "Left" or "Right"

    # ── Machine-readable format: HAND <side> <48 floats> ──────────────────
    # Layout: cx cy | kp0x kp0y ... kp20x kp20y | bx1 by1 bx2 by2
    numeric_tokens = tokens[2:]
    try:
        vals = np.array([float(v) for v in numeric_tokens], dtype=np.float32)
        if vals.size == 48:
            centroid  = vals[0:2]
            kp_flat   = vals[2:44]          # 21 * 2 = 42 values
            bbox_vals = vals[44:48]
            keypoints = kp_flat.reshape(21, 2)
            bbox = (int(bbox_vals[0]), int(bbox_vals[1]),
                    int(bbox_vals[2]), int(bbox_vals[3]))
            return HandSnapshot(
                handedness=handedness,
                centroid=centroid,
                keypoints=keypoints,
                bbox=bbox,
            )
    except (ValueError, IndexError):
        pass

    # centroid=(<cx>, <cy>)
    m_c = re_centroid.search(raw_line)
    m_b = re_bbox.search(raw_line)
    if m_c is None or m_b is None:
        return None

    cx, cy = float(m_c.group(1)), float(m_c.group(2))
    x1, y1 = int(m_b.group(1)), int(m_b.group(2))
    x2, y2 = int(m_b.group(3)), int(m_b.group(4))

    centroid = np.array([cx, cy], dtype=np.float32)
    keypoints = _synthetic_keypoints(x1, y1, x2, y2)

    return HandSnapshot(
        handedness=handedness,
        centroid=centroid,
        keypoints=keypoints,
        bbox=(x1, y1, x2, y2),
    )


def _parse_object_line(tokens: List[str]) -> Optional[ObjectSnapshot]:
    """
    Parse an OBJECT line.

    Supports:
        OBJECT <label> <cx> <cy> <bx1> <by1> <bx2> <by2>          (7 tokens)
        OBJECT  <label>  centroid=(<cx>, <cy>)  bbox=(...)          (keyword)
    """
    if len(tokens) < 3:
        return None

    label = tokens[1]

    # Keyword format
    raw = " ".join(tokens)
    m_c = re.search(r"centroid=\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", raw)
    m_b = re.search(r"bbox=\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", raw)
    if m_c and m_b:
        cx, cy = float(m_c.group(1)), float(m_c.group(2))
        x1, y1 = int(m_b.group(1)), int(m_b.group(2))
        x2, y2 = int(m_b.group(3)), int(m_b.group(4))
        return ObjectSnapshot(
            label=label,
            centroid=np.array([cx, cy], dtype=np.float32),
            bbox=(x1, y1, x2, y2),
        )

    # Positional format: OBJECT <label> <cx> <cy> <bx1> <by1> <bx2> <by2>
    try:
        vals = np.array([float(v) for v in tokens[2:]], dtype=np.float32)
        if vals.size >= 6:
            centroid = vals[0:2]
            bbox = (int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]))
            return ObjectSnapshot(label=label, centroid=centroid, bbox=bbox)
    except (ValueError, IndexError):
        pass

    return None


def _synthetic_keypoints(
    x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Build a (21, 2) keypoint array from a bounding box when the actual
    MediaPipe landmarks are unavailable in the tracking file.

    The 21 points are placed on a regular 3×7 grid inside the bounding box.
    Index 0 (wrist) is positioned at the bottom-center of the box.
    """
    kp = np.zeros((21, 2), dtype=np.float32)
    # Wrist at bottom-center
    kp[0] = [(x1 + x2) / 2.0, float(y2)]
    # Remaining 20 keypoints on a grid
    xs = np.linspace(x1, x2, 5)
    ys = np.linspace(y1, y2, 4)
    idx = 1
    for y in ys:
        for x in xs:
            if idx >= 21:
                break
            kp[idx] = [x, y]
            idx += 1
    return kp


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic pre-computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dt(records: List[FrameRecord], t: int, fps: float) -> float:
    """Time step in seconds between frame t-1 and frame t."""
    if t == 0:
        return 1.0 / fps
    dt = (records[t].timestamp_ms - records[t - 1].timestamp_ms) / 1000.0
    return dt if dt > 0 else 1.0 / fps


def _fill_positions(raw: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaN positions in a (T, 2) array."""
    arr = raw.copy()
    last = None
    for t in range(len(arr)):
        if not np.isnan(arr[t, 0]):
            last = arr[t].copy()
        elif last is not None:
            arr[t] = last
    last = None
    for t in range(len(arr) - 1, -1, -1):
        if not np.isnan(arr[t, 0]):
            last = arr[t].copy()
        elif last is not None:
            arr[t] = last
    return arr


def compute_hand_positions(
    records:  List[FrameRecord],
    use_hand2: bool = False,
) -> np.ndarray:
    """
    Build the array p_h(t) of shape (T, 2).
    """
    T  = len(records)
    ph = np.full((T, 2), np.nan, dtype=np.float64)
    for t, rec in enumerate(records):
        src = rec.hand2 if use_hand2 else rec.hand
        if src is not None:
            ph[t] = src.centroid
    return _fill_positions(ph)


def compute_hand_velocities(
    ph:      np.ndarray,
    records: List[FrameRecord],
    fps:     float,
) -> np.ndarray:
    """
    v_h(t) = (p_h(t) - p_h(t-1)) / dt,  shape (T, 2).
    Frame 0 has zero velocity.
    """
    T  = len(records)
    vh = np.zeros((T, 2), dtype=np.float64)
    for t in range(1, T):
        dt = _dt(records, t, fps)
        vh[t] = (ph[t] - ph[t - 1]) / dt
    return vh


def compute_object_positions(
    records:    List[FrameRecord],
    obj_labels: List[str],
) -> Dict[str, np.ndarray]:
    """
    Build p_oi(t) arrays, shape (T, 2), one per object label.
    Missing detections are forward/backward filled.
    """
    T = len(records)
    pos: Dict[str, np.ndarray] = {}
    for label in obj_labels:
        arr = np.full((T, 2), np.nan, dtype=np.float64)
        for t, rec in enumerate(records):
            if label in rec.objects:
                arr[t] = rec.objects[label].centroid
        pos[label] = _fill_positions(arr)
    return pos


def collect_object_labels(records: List[FrameRecord]) -> List[str]:
    """Return sorted list of all unique object labels across all frames."""
    labels: set = set()
    for rec in records:
        labels.update(rec.objects.keys())
    return sorted(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Percentiles
# ─────────────────────────────────────────────────────────────────────────────

def _p90(values: np.ndarray) -> float:
    """90th percentile; falls back to 1.0 to avoid division by zero."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 90)) if len(v) > 0 and np.percentile(v, 90) > 0 else 1.0


def _p10(values: np.ndarray) -> float:
    """10th percentile; falls back to a small positive value."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.percentile(v, 10)) if len(v) > 0 else 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Cost term functions
# Each function returns a 1-D numpy array of length T (one value per frame).
# ─────────────────────────────────────────────────────────────────────────────

def distance_cost(
    ph:    np.ndarray,
    poi:   np.ndarray,
    d_max: float,
) -> np.ndarray:
    """
    phi_d_i(t) = ||p_h(t) - p_oi(t)|| / d_max  (Eq. 2)
    """
    dist = np.linalg.norm(ph - poi, axis=1)
    return dist / (d_max + 1e-9)


def hand_velocity_cost(
    vh:     np.ndarray,
    vh_max: float,
) -> np.ndarray:
    """
    phi_v(t) = ||v_h(t)|| / vh_max  (Eq. 3)
    """
    speed = np.linalg.norm(vh, axis=1)
    return speed / (vh_max + 1e-9)


def hand_direction_cost(
    vh:  np.ndarray,
    ph:  np.ndarray,
    poi: np.ndarray,
) -> np.ndarray:
    """
    phi_dir_i(t) = 1 - max(0, cos(theta_i(t)))  (Eq. 4)
    """
    diff  = ph - poi
    speed = np.linalg.norm(vh,   axis=1, keepdims=True)
    dist  = np.linalg.norm(diff, axis=1, keepdims=True)

    denom = speed * dist
    valid = (denom[:, 0] > 1e-9)

    cos_theta        = np.ones(len(vh), dtype=np.float64)
    cos_theta[valid] = (
        np.sum(vh[valid] * diff[valid], axis=1) / denom[valid, 0]
    )
    return 1.0 - np.maximum(0.0, cos_theta)


def object_velocity_cost(
    ph:      np.ndarray,
    poi:     np.ndarray,
    records: List[FrameRecord],
    fps:     float,
    v_th:    float,
) -> np.ndarray:
    """
    Relative hand-object velocity cost  (Eq. 5)
    """
    T   = len(records)
    di  = np.linalg.norm(ph - poi, axis=1)
    voi = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        dt     = _dt(records, t, fps)
        voi[t] = abs(di[t] - di[t - 1]) / dt

    phi_obj = np.where(
        np.abs(voi) < v_th,
        0.0,
        (np.abs(voi) - v_th) / (v_th ** 2 + 1e-9),
    )
    return phi_obj


def hand_compactness_cost(
    records:  List[FrameRecord],
    poi:      np.ndarray,
    R_ref:    float,
    sigma_d:  float,
    use_hand2: bool = False,
) -> np.ndarray:
    """
    phi_comp(t) = 1 / (1 + w_d(t) * alpha(t))  (Eq. 8)

    Parameters
    ----------
    use_hand2 : select secondary hand keypoints when True
    """
    T        = len(records)
    phi_comp = np.ones(T, dtype=np.float64)

    for t, rec in enumerate(records):
        src = rec.hand2 if use_hand2 else rec.hand
        if src is None:
            continue

        R_t   = src.hand_spread()
        alpha = 1.0 - R_t / (R_ref + 1e-9)

        fingertips = src.fingertips
        obj_pos    = poi[t]
        diffs      = fingertips - obj_pos[np.newaxis, :]
        d_min      = float(np.min(np.linalg.norm(diffs, axis=1)))
        w_d        = np.exp(-(d_min ** 2) / (2.0 * sigma_d ** 2 + 1e-9))

        phi_comp[t] = 1.0 / (1.0 + w_d * alpha)

    return phi_comp


def enclosure_cost(
    records:   List[FrameRecord],
    poi:       np.ndarray,
    use_hand2: bool = False,
) -> np.ndarray:
    """
    phi_enc_i(t) = exp(-(rho_i(t) - 1)) - 1  (Eq. 10)

    Parameters
    ----------
    use_hand2 : select secondary hand keypoints when True
    """
    from scipy.spatial import ConvexHull, Delaunay

    T       = len(records)
    phi_enc = np.full(T, np.exp(1.0) - 1.0, dtype=np.float64)

    for t, rec in enumerate(records):
        src = rec.hand2 if use_hand2 else rec.hand
        if src is None:
            continue

        kp = src.keypoints   # (21, 2)

        obj_label = _find_object_label_for_frame(rec)
        if obj_label is None:
            continue
        obj = rec.objects[obj_label]
        bx1, by1, bx2, by2 = obj.bbox
        test_pts = np.array([
            [bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2],
            [(bx1 + bx2) / 2, (by1 + by2) / 2],
        ], dtype=np.float64)

        try:
            hull   = ConvexHull(kp)
            tri    = Delaunay(kp[hull.vertices])
            inside = tri.find_simplex(test_pts) >= 0
            rho    = float(np.mean(inside))
        except Exception:
            rho = 0.0

        phi_enc[t] = np.exp(-(rho - 1.0)) - 1.0

    return phi_enc


def coupling_term(
    phi_d: np.ndarray,
    phi_v: np.ndarray,
) -> np.ndarray:
    """
    phi_couple_i(t) = max(0, exp(phi_d_i(t) * phi_v(t)) - 1)  (Eq. 11)
    """
    return np.maximum(0.0, np.exp(phi_d * phi_v) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _find_object_label_for_frame(rec: FrameRecord) -> Optional[str]:
    """Return the first object label present in a frame, or None."""
    return next(iter(rec.objects), None)


# ─────────────────────────────────────────────────────────────────────────────
# Global normalisation constants  (computed once across all objects and frames)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_normalisation(
    ph:       np.ndarray,
    vh:       np.ndarray,
    obj_pos:  Dict[str, np.ndarray],
    records:  List[FrameRecord],
    fps:      float,
    # Optional second-hand arrays; ignored when None
    ph2:      Optional[np.ndarray] = None,
    vh2:      Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all P90 / P10 constants needed for normalisation.

    When a second hand is present (ph2, vh2 are not None) the statistics are
    pooled over both hands so that a single consistent normalisation is used
    for both cost evaluations.

    Returns a dict with keys:
      d_max    – P90 of all hand-object distances across all objects/frames
      vh_max   – P90 of hand speed
      v_th     – P10 of all relative hand-object velocities
      R_ref    – P90 of hand spread R(t)
      sigma_d  – P90 of all fingertip-to-object minimum distances
    """
    T = len(records)

    # Helper to collect per-hand relative velocities for a given ph array
    def _vrel_for_hand(ph_arr: np.ndarray) -> List[np.ndarray]:
        segments = []
        for poi in obj_pos.values():
            di  = np.linalg.norm(ph_arr - poi, axis=1)
            voi = np.zeros(T, dtype=np.float64)
            for t in range(1, T):
                dt     = _dt(records, t, fps)
                voi[t] = abs(di[t] - di[t - 1]) / dt
            segments.append(voi[1:])
        return segments

    # Helper to collect fingertip-to-object min distances for a given hand slot
    def _dmin_for_hand(use_hand2: bool) -> List[float]:
        dmin_vals = []
        for label, poi in obj_pos.items():
            for t, rec in enumerate(records):
                src = rec.hand2 if use_hand2 else rec.hand
                if src is None:
                    continue
                diffs = src.fingertips - poi[t][np.newaxis, :]
                dmin_vals.append(float(np.min(np.linalg.norm(diffs, axis=1))))
        return dmin_vals

    # ── d_max ────────────────────────────────────────────────────────────────
    all_dists = []
    for poi in obj_pos.values():
        all_dists.append(np.linalg.norm(ph - poi, axis=1))
        if ph2 is not None:
            all_dists.append(np.linalg.norm(ph2 - poi, axis=1))
    d_max = _p90(np.concatenate(all_dists)) if all_dists else 1.0

    # ── vh_max ───────────────────────────────────────────────────────────────
    speeds = [np.linalg.norm(vh, axis=1)]
    if vh2 is not None:
        speeds.append(np.linalg.norm(vh2, axis=1))
    vh_max = _p90(np.concatenate(speeds))

    # ── v_th ─────────────────────────────────────────────────────────────────
    all_vrel = _vrel_for_hand(ph)
    if ph2 is not None:
        all_vrel += _vrel_for_hand(ph2)
    v_th = _p10(np.concatenate(all_vrel)) if all_vrel else 1e-6

    # ── R_ref ─────────────────────────────────────────────────────────────────
    R_vals = []
    for rec in records:
        if rec.hand is not None:
            R_vals.append(rec.hand.hand_spread())
        if rec.hand2 is not None:
            R_vals.append(rec.hand2.hand_spread())
    R_ref = _p90(np.array(R_vals)) if R_vals else 1.0

    # ── sigma_d ───────────────────────────────────────────────────────────────
    dmin_vals = _dmin_for_hand(use_hand2=False)
    if ph2 is not None:
        dmin_vals += _dmin_for_hand(use_hand2=True)
    sigma_d = _p90(np.array(dmin_vals)) if dmin_vals else 1.0

    return dict(d_max=d_max, vh_max=vh_max, v_th=v_th,
                R_ref=R_ref, sigma_d=sigma_d)


# ─────────────────────────────────────────────────────────────────────────────
# Per-object cost evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_costs_for_object(
    label:     str,
    ph:        np.ndarray,
    vh:        np.ndarray,
    poi:       np.ndarray,
    records:   List[FrameRecord],
    fps:       float,
    norms:     dict,
    use_hand2: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute all seven cost terms and J_i(t) for a single object i and a
    single hand.

    Parameters
    ----------
    use_hand2 : route gesture-based terms (compactness, enclosure) to the
                secondary hand when True.

    Returns a dict mapping term names to 1-D arrays of length T.
    """
    d_max   = norms["d_max"]
    vh_max  = norms["vh_max"]
    v_th    = norms["v_th"]
    R_ref   = norms["R_ref"]
    sigma_d = norms["sigma_d"]

    phi_d      = distance_cost(ph, poi, d_max)
    phi_v      = hand_velocity_cost(vh, vh_max)
    phi_dir    = hand_direction_cost(vh, ph, poi)
    phi_obj    = object_velocity_cost(ph, poi, records, fps, v_th)
    phi_comp   = hand_compactness_cost(records, poi, R_ref, sigma_d, use_hand2=use_hand2)
    phi_enc    = enclosure_cost(records, poi, use_hand2=use_hand2)
    phi_couple = coupling_term(phi_d, phi_v)

    J = phi_d + phi_v + phi_dir + phi_obj + phi_comp + phi_enc + phi_couple

    return {
        "phi_d":       phi_d,
        "phi_v":       phi_v,
        "phi_dir":     phi_dir,
        "phi_obj":     phi_obj,
        "phi_comp":    phi_comp,
        "phi_enc":     phi_enc,
        "phi_couple":  phi_couple,
        "J":           J,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def save_cost_results(
    output_path: str,
    records:     List[FrameRecord],
    obj_labels:  List[str],
    costs:       Dict[str, Dict[str, np.ndarray]],
    fps:         float,
    num_hands:   int = 1,
) -> None:
    """
    Write cost time-histories for every object to a formatted .txt file.

    File structure
    --------------
    A header section followed by one DATA BLOCK per object.

    Single-hand mode (num_hands == 1)
    -------------------------------------
    Each block contains columns:
      frame_idx  timestamp_ms  phi_d  phi_v  phi_dir  phi_obj
                 phi_comp  phi_enc  phi_couple  J

    Two-hand mode (num_hands == 2)
    -------------------------------------
    Each block contains all hand-1 terms, all hand-2 terms, and then the
    averaged final cost J_avg:

      frame_idx  timestamp_ms
        h1_phi_d  h1_phi_v  h1_phi_dir  h1_phi_obj  h1_phi_comp  h1_phi_enc  h1_phi_couple  h1_J
        h2_phi_d  h2_phi_v  h2_phi_dir  h2_phi_obj  h2_phi_comp  h2_phi_enc  h2_phi_couple  h2_J
        J_avg

    Column widths are fixed for easy parsing with numpy.loadtxt or pandas.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    TERM_NAMES = ["phi_d", "phi_v", "phi_dir", "phi_obj",
                  "phi_comp", "phi_enc", "phi_couple", "J"]
    COL_W = 14   # fixed column width

    def _fmt(x: float) -> str:
        return f"{x:{COL_W}.6f}"

    with open(output_path, "w") as fh:

        # ── File header ───────────────────────────────────────────────────────
        fh.write("# ============================================================\n")
        fh.write("# Hand-Object Interaction Cost Function  —  time history\n")
        fh.write("#\n")
        fh.write(f"# Tracking source : {TRACKING_RESULTS_PATH}\n")
        fh.write(f"# Total frames    : {len(records)}\n")
        fh.write(f"# FPS             : {fps:.2f}\n")
        fh.write(f"# num_hands       : {num_hands}\n")
        fh.write(f"# Tracked objects : {', '.join(obj_labels)}\n")
        fh.write("#\n")
        fh.write("# Cost terms (paper equations):\n")
        fh.write("#   phi_d      – distance cost              (Eq. 2)\n")
        fh.write("#   phi_v      – hand velocity cost         (Eq. 3)\n")
        fh.write("#   phi_dir    – hand direction cost        (Eq. 4)\n")
        fh.write("#   phi_obj    – object velocity cost       (Eq. 5)\n")
        fh.write("#   phi_comp   – hand compactness cost      (Eq. 8)\n")
        fh.write("#   phi_enc    – enclosure cost             (Eq. 10)\n")
        fh.write("#   phi_couple – coupling term              (Eq. 11)\n")
        fh.write("#   J          – total cost per hand        (Eq. 12)\n")
        if num_hands == 2:
            fh.write("#   J_avg      – average of J_hand1 and J_hand2\n")
        fh.write("# ============================================================\n\n")

        for label in obj_labels:
            fh.write(f"# {'=' * 58}\n")
            fh.write(f"# OBJECT: {label}\n")
            fh.write(f"# {'=' * 58}\n")

            if num_hands == 1:
                # ── Single-hand block ─────────────────────────────────────────
                obj_costs = costs[label]
                col_names = ["frame_idx", "timestamp_ms"] + TERM_NAMES
                header_parts = [f"{n:>{COL_W}}" for n in col_names]
                fh.write("#" + "".join(header_parts) + "\n")

                for t in range(len(records)):
                    row = [
                        f"{records[t].frame_idx:{COL_W}d}",
                        f"{records[t].timestamp_ms:{COL_W}.3f}",
                    ]
                    row += [_fmt(float(obj_costs[term][t])) for term in TERM_NAMES]
                    fh.write(" ".join(row) + "\n")

            else:
                # ── Two-hand block ────────────────────────────────────────────
                h1_costs  = costs[f"{label}__hand1"]
                h2_costs  = costs[f"{label}__hand2"]
                J_avg     = costs[f"{label}__J_avg"]["J_avg"]

                # Prefixed column names
                h1_cols = [f"h1_{t}" for t in TERM_NAMES]
                h2_cols = [f"h2_{t}" for t in TERM_NAMES]
                col_names = (["frame_idx", "timestamp_ms"]
                             + h1_cols + h2_cols + ["J_avg"])
                header_parts = [f"{n:>{COL_W}}" for n in col_names]
                fh.write("#" + "".join(header_parts) + "\n")

                for t in range(len(records)):
                    row = [
                        f"{records[t].frame_idx:{COL_W}d}",
                        f"{records[t].timestamp_ms:{COL_W}.3f}",
                    ]
                    row += [_fmt(float(h1_costs[term][t])) for term in TERM_NAMES]
                    row += [_fmt(float(h2_costs[term][t])) for term in TERM_NAMES]
                    row.append(_fmt(float(J_avg[t])))
                    fh.write(" ".join(row) + "\n")

            fh.write("\n")   # blank line between object blocks

    print(f"[Writer] Cost results saved to '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_costs(
    tracking_path: str,
    output_path:   str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    """
    Full cost-function evaluation pipeline.

    Returns
    -------
    costs     : dict keyed by object label (single-hand mode) or by
                "<label>__hand1", "<label>__hand2", "<label>__J_avg"
                (two-hand mode)
    num_hands : number of hands detected in the tracking file (1 or 2)
    """
    # ── Load tracking data ─────────────────────────────────────────────────
    records, fps, num_hands = parse_tracking_file(tracking_path)
    if not records:
        raise RuntimeError("No frame records found in the tracking file.")

    obj_labels = collect_object_labels(records)
    if not obj_labels:
        raise RuntimeError("No objects found in the tracking file.")

    print(f"[Cost]   Objects detected : {obj_labels}")
    print(f"[Cost]   num_hands        : {num_hands}")

    # ── Kinematic arrays — hand 1 ─────────────────────────────────────────
    ph1 = compute_hand_positions(records, use_hand2=False)
    vh1 = compute_hand_velocities(ph1, records, fps)

    # ── Kinematic arrays — hand 2 (when present) ──────────────────────────
    ph2: Optional[np.ndarray] = None
    vh2: Optional[np.ndarray] = None
    if num_hands == 2:
        ph2 = compute_hand_positions(records, use_hand2=True)
        vh2 = compute_hand_velocities(ph2, records, fps)

    obj_pos = compute_object_positions(records, obj_labels)

    # ── Global normalisation constants (pooled over both hands if present) ─
    norms = compute_global_normalisation(ph1, vh1, obj_pos, records, fps,
                                         ph2=ph2, vh2=vh2)
    print("[Cost]   Normalisation constants:")
    for k, v in norms.items():
        print(f"           {k:10s} = {v:.4f}")

    # ── Cost evaluation ────────────────────────────────────────────────────
    all_costs: Dict[str, Dict[str, np.ndarray]] = {}

    for label in obj_labels:
        poi = obj_pos[label]

        if num_hands == 1:
            # Single-hand path: store costs directly under the object label
            print(f"[Cost]   Computing costs for '{label}' (hand 1) ...")
            all_costs[label] = evaluate_costs_for_object(
                label=label, ph=ph1, vh=vh1, poi=poi,
                records=records, fps=fps, norms=norms, use_hand2=False,
            )

        else:
            # Two-hand path
            print(f"[Cost]   Computing costs for '{label}' (hand 1) ...")
            h1_costs = evaluate_costs_for_object(
                label=label, ph=ph1, vh=vh1, poi=poi,
                records=records, fps=fps, norms=norms, use_hand2=False,
            )

            print(f"[Cost]   Computing costs for '{label}' (hand 2) ...")
            h2_costs = evaluate_costs_for_object(
                label=label, ph=ph2, vh=vh2, poi=poi,
                records=records, fps=fps, norms=norms, use_hand2=True,
            )

            # Final cost: element-wise average of J from both hands
            J_avg = 0.5 * (h1_costs["J"] + h2_costs["J"])

            all_costs[f"{label}__hand1"] = h1_costs
            all_costs[f"{label}__hand2"] = h2_costs
            all_costs[f"{label}__J_avg"] = {"J_avg": J_avg}

    save_cost_results(output_path, records, obj_labels, all_costs, fps, num_hands)

    return all_costs, num_hands


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    costs, num_hands = compute_all_costs(TRACKING_RESULTS_PATH, COST_OUTPUT_PATH)

    print("\n[Summary]  Mean cost values across all frames:")

    if num_hands == 1:
        print(f"  {'Object':<25s}  {'mean J':>10s}  {'min J':>10s}  {'max J':>10s}")
        print("  " + "-" * 60)
        for label, obj_costs in costs.items():
            J = obj_costs["J"]
            print(f"  {label:<25s}  {np.mean(J):10.4f}  "
                  f"{np.min(J):10.4f}  {np.max(J):10.4f}")
    else:
        # Collect unique object labels from the two-hand cost keys
        obj_labels = sorted({
            k.replace("__hand1", "").replace("__hand2", "").replace("__J_avg", "")
            for k in costs
        })
        print(f"  {'Object':<25s}  {'mean J_h1':>10s}  {'mean J_h2':>10s}  "
              f"{'mean J_avg':>10s}  {'min J_avg':>10s}  {'max J_avg':>10s}")
        print("  " + "-" * 80)
        for label in obj_labels:
            J_h1  = costs[f"{label}__hand1"]["J"]
            J_h2  = costs[f"{label}__hand2"]["J"]
            J_avg = costs[f"{label}__J_avg"]["J_avg"]
            print(f"  {label:<25s}  {np.mean(J_h1):10.4f}  {np.mean(J_h2):10.4f}  "
                  f"{np.mean(J_avg):10.4f}  {np.min(J_avg):10.4f}  {np.max(J_avg):10.4f}")

    print("\n[DONE]")