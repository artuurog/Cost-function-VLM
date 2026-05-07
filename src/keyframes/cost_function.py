"""
cost_function.py
================
Evaluate the hand-object interaction cost terms
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


TRACKING_RESULTS_PATH = "results/tracking_results.txt"
COST_OUTPUT_PATH      = "results/cost_function.txt"

# Fingertip keypoint indices (MediaPipe convention)
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
    """All tracking data for a single video frame."""
    frame_idx:    int
    timestamp_ms: float
    hand:         Optional[HandSnapshot]          # None when hand not detected
    objects:      Dict[str, ObjectSnapshot] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking results parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_tracking_file(path: str) -> Tuple[List[FrameRecord], float]:
    """
    Read a tracking-results .txt file and return:
      - list of FrameRecord (one per frame)
      - video FPS (parsed from header comment, default 30.0)

    """
    records: List[FrameRecord] = []
    fps = 30.0
    current: Optional[FrameRecord] = None

    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # ── Header comments ───────────────────────────────────────────
            if line.startswith("#"):
                m = re.search(r"FPS[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if m:
                    fps = float(m.group(1))
                continue

            tokens = line.split()
            tag = tokens[0].upper()

            # ── FRAME line ────────────────────────────────────────────────
            if tag == "FRAME":
                # Store previous frame before starting a new one
                if current is not None:
                    records.append(current)
                frame_idx    = int(tokens[1])
                timestamp_ms = float(tokens[2])
                current = FrameRecord(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    hand=None,
                )

            # ── HAND line ─────────────────────────────────────────────────
            elif tag == "HAND" and current is not None:
                # tokens: HAND <handedness> <48 floats>
                handedness = tokens[1]
                vals = np.array([float(v) for v in tokens[2:]], dtype=np.float32)
                # Layout: cx cy | kp0x kp0y ... kp20x kp20y | bx1 by1 bx2 by2
                centroid  = vals[0:2]
                kp_flat   = vals[2:44]          # 21 * 2 = 42 values
                bbox_vals = vals[44:48]
                keypoints = kp_flat.reshape(21, 2)
                bbox = (int(bbox_vals[0]), int(bbox_vals[1]),
                        int(bbox_vals[2]), int(bbox_vals[3]))
                current.hand = HandSnapshot(
                    handedness=handedness,
                    centroid=centroid,
                    keypoints=keypoints,
                    bbox=bbox,
                )

            # ── OBJECT line ───────────────────────────────────────────────
            elif tag == "OBJECT" and current is not None:
                # tokens: OBJECT <label> <cx> <cy> <bx1> <by1> <bx2> <by2>
                label = tokens[1]
                vals  = np.array([float(v) for v in tokens[2:]], dtype=np.float32)
                centroid = vals[0:2]
                bbox = (int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]))
                current.objects[label] = ObjectSnapshot(
                    label=label, centroid=centroid, bbox=bbox
                )

    # Append the last frame
    if current is not None:
        records.append(current)

    print(f"[Parser] Loaded {len(records)} frames from '{path}'  (fps={fps})")
    return records, fps


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic pre-computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dt(records: List[FrameRecord], t: int, fps: float) -> float:
    """Time step in seconds between frame t-1 and frame t."""
    if t == 0:
        return 1.0 / fps
    dt = (records[t].timestamp_ms - records[t - 1].timestamp_ms) / 1000.0
    return dt if dt > 0 else 1.0 / fps


def compute_hand_positions(records: List[FrameRecord]) -> np.ndarray:
    """
    Build the array p_h(t) of shape (T, 2).
    Frames with no detected hand are filled by forward/backward propagation.
    """
    T  = len(records)
    ph = np.full((T, 2), np.nan, dtype=np.float64)
    for t, rec in enumerate(records):
        if rec.hand is not None:
            ph[t] = rec.hand.centroid
    # Forward-fill NaNs
    last_valid = None
    for t in range(T):
        if not np.isnan(ph[t, 0]):
            last_valid = ph[t].copy()
        elif last_valid is not None:
            ph[t] = last_valid
    # Backward-fill any leading NaNs
    last_valid = None
    for t in range(T - 1, -1, -1):
        if not np.isnan(ph[t, 0]):
            last_valid = ph[t].copy()
        elif last_valid is not None:
            ph[t] = last_valid
    return ph


def compute_hand_velocities(
    ph: np.ndarray,
    records: List[FrameRecord],
    fps: float,
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
    records: List[FrameRecord],
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
        # Forward fill
        last = None
        for t in range(T):
            if not np.isnan(arr[t, 0]):
                last = arr[t].copy()
            elif last is not None:
                arr[t] = last
        # Backward fill
        last = None
        for t in range(T - 1, -1, -1):
            if not np.isnan(arr[t, 0]):
                last = arr[t].copy()
            elif last is not None:
                arr[t] = last
        pos[label] = arr
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
    ph:        np.ndarray,
    poi:       np.ndarray,
    d_max:     float,
) -> np.ndarray:
    """
    phi_d_i(t) = ||p_h(t) - p_oi(t)|| / d_max

    Parameters
    ----------
    ph    : hand centroid positions, shape (T, 2)
    poi   : object i centroid positions, shape (T, 2)
    d_max : robust normalisation constant (P90 of all distances)

    Returns
    -------
    phi_d : shape (T,)
    """
    dist  = np.linalg.norm(ph - poi, axis=1)           # (T,)
    return dist / (d_max + 1e-9)


def hand_velocity_cost(
    vh:      np.ndarray,
    vh_max:  float,
) -> np.ndarray:
    """
    phi_v(t) = ||v_h(t)|| / vh_max

    Parameters
    ----------
    vh     : hand velocity, shape (T, 2)
    vh_max : robust normalisation constant (P90 of hand speed)

    Returns
    -------
    phi_v : shape (T,)
    """
    speed = np.linalg.norm(vh, axis=1)                 # (T,)
    return speed / (vh_max + 1e-9)


def hand_direction_cost(
    vh:  np.ndarray,
    ph:  np.ndarray,
    poi: np.ndarray,
) -> np.ndarray:
    """
    phi_dir_i(t) = 1 - max(0, cos(theta_i(t)))

    cos(theta_i) = v_h · (p_h - p_oi) / (||v_h|| * ||p_h - p_oi||)

    When hand speed or distance is zero the cosine is undefined;
    the term defaults to 1 (neutral / high cost) in those cases.

    Parameters
    ----------
    vh  : hand velocity, shape (T, 2)
    ph  : hand centroid, shape (T, 2)
    poi : object i centroid, shape (T, 2)

    Returns
    -------
    phi_dir : shape (T,)
    """
    diff      = ph - poi                                # (T, 2)  hand → object vector
    speed     = np.linalg.norm(vh,   axis=1, keepdims=True)   # (T, 1)
    dist      = np.linalg.norm(diff, axis=1, keepdims=True)   # (T, 1)

    denom = speed * dist
    valid = (denom[:, 0] > 1e-9)

    cos_theta        = np.ones(len(vh), dtype=np.float64)
    cos_theta[valid] = (
        np.sum(vh[valid] * diff[valid], axis=1)
        / denom[valid, 0]
    )

    return 1.0 - np.maximum(0.0, cos_theta)


def object_velocity_cost(
    ph:    np.ndarray,
    poi:   np.ndarray,
    records: List[FrameRecord],
    fps:   float,
    v_th:  float,
) -> np.ndarray:
    """
    Relative hand-object velocity cost

    v_oi(t) = d/dt ||p_h(t) - p_oi(t)||
    phi_obj_i(t) = 0                            if ||v_oi|| < v_th
                 = (||v_oi|| - v_th) / v_th^2   otherwise

    Parameters
    ----------
    ph      : hand centroid, shape (T, 2)
    poi     : object i centroid, shape (T, 2)
    records : list of FrameRecord (for time-step computation)
    fps     : video frame rate
    v_th    : threshold = P10 of all relative velocities (pre-computed globally)

    Returns
    -------
    phi_obj : shape (T,)
    """
    T   = len(records)
    di  = np.linalg.norm(ph - poi, axis=1)             # hand-object distance (T,)
    voi = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        dt      = _dt(records, t, fps)
        voi[t]  = abs(di[t] - di[t - 1]) / dt

    phi_obj = np.where(
        np.abs(voi) < v_th,
        0.0,
        (np.abs(voi) - v_th) / (v_th ** 2 + 1e-9),
    )
    return phi_obj


def hand_compactness_cost(
    records:    List[FrameRecord],
    poi:        np.ndarray,
    R_ref:      float,
    sigma_d:    float,
) -> np.ndarray:
    """
    phi_comp(t) = 1 / (1 + w_d(t) * alpha(t))

    with
      R(t)     = mean distance of each keypoint from wrist
      alpha(t) = 1 - R(t) / R_ref
      d_min_i(t) = min distance from fingertips to object i
      w_d(t)   = exp( -d_min_i(t)^2 / (2 * sigma_d^2) )
    Parameters
    ----------
    records  : list of FrameRecord
    poi      : object i centroid positions, shape (T, 2)
    R_ref    : P90 of R(t) across the video (pre-computed globally)
    sigma_d  : P90 of d_min(t) across all objects and frames

    Returns
    -------
    phi_comp : shape (T,)
    """
    T        = len(records)
    phi_comp = np.ones(T, dtype=np.float64)            # default: no compactness

    for t, rec in enumerate(records):
        if rec.hand is None:
            continue

        R_t    = rec.hand.hand_spread()
        alpha  = 1.0 - R_t / (R_ref + 1e-9)

        # Minimum fingertip-to-object distance
        fingertips  = rec.hand.fingertips            # (5, 2)
        obj_pos     = poi[t]                         # (2,)
        diffs       = fingertips - obj_pos[np.newaxis, :]
        d_min       = float(np.min(np.linalg.norm(diffs, axis=1)))

        w_d = np.exp(-(d_min ** 2) / (2.0 * sigma_d ** 2 + 1e-9))

        phi_comp[t] = 1.0 / (1.0 + w_d * alpha)

    return phi_comp


def enclosure_cost(
    records: List[FrameRecord],
    poi:     np.ndarray,
) -> np.ndarray:
    """
    phi_enc_i(t) = exp(-(rho_i(t) - 1)) - 1

    Parameters
    ----------
    records : list of FrameRecord
    poi     : object i centroid positions, shape (T, 2)  (used as fallback)

    Returns
    -------
    phi_enc : shape (T,)
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import Delaunay                 # for point-in-hull test

    T       = len(records)
    phi_enc = np.full(T, np.exp(1.0) - 1.0, dtype=np.float64)   # worst case

    for t, rec in enumerate(records):
        if rec.hand is None:
            continue

        kp = rec.hand.keypoints                      # (21, 2)

        # Collect object bounding box sample points (corners + centroid)
        obj_label = _find_object_label_for_frame(rec)
        if obj_label is None:
            continue
        obj   = rec.objects[obj_label]
        bx1, by1, bx2, by2 = obj.bbox
        test_pts = np.array([
            [bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2],   # corners
            [(bx1 + bx2) / 2, (by1 + by2) / 2],                # centroid
        ], dtype=np.float64)

        try:
            hull  = ConvexHull(kp)
            tri   = Delaunay(kp[hull.vertices])
            inside = tri.find_simplex(test_pts) >= 0
            rho   = float(np.mean(inside))
        except Exception:
            rho = 0.0

        phi_enc[t] = np.exp(-(rho - 1.0)) - 1.0

    return phi_enc


def coupling_term(
    phi_d: np.ndarray,
    phi_v: np.ndarray,
) -> np.ndarray:
    """
    phi_couple_i(t) = max(0, exp(phi_d_i(t) * phi_v(t)) - 1)

    Parameters
    ----------
    phi_d : distance cost,     shape (T,)
    phi_v : hand velocity cost, shape (T,)

    Returns
    -------
    phi_couple : shape (T,)
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
    ph:          np.ndarray,
    vh:          np.ndarray,
    obj_pos:     Dict[str, np.ndarray],
    records:     List[FrameRecord],
    fps:         float,
) -> dict:
    """
    Compute all P90 / P10 constants needed for normalisation.

    Returns a dict with keys:
      d_max    – P90 of all hand-object distances across all objects/frames
      vh_max   – P90 of hand speed
      v_th     – P10 of all relative hand-object velocities
      R_ref    – P90 of hand spread R(t)
      sigma_d  – P90 of all fingertip-to-object minimum distances
    """
    T = len(records)

    # ── d_max ────────────────────────────────────────────────────────────────
    all_dists = []
    for poi in obj_pos.values():
        all_dists.append(np.linalg.norm(ph - poi, axis=1))
    d_max = _p90(np.concatenate(all_dists)) if all_dists else 1.0

    # ── vh_max ───────────────────────────────────────────────────────────────
    speeds = np.linalg.norm(vh, axis=1)
    vh_max = _p90(speeds)

    # ── v_th (P10 of relative velocities) ────────────────────────────────────
    all_vrel = []
    for poi in obj_pos.values():
        di = np.linalg.norm(ph - poi, axis=1)
        voi = np.zeros(T, dtype=np.float64)
        for t in range(1, T):
            dt = _dt(records, t, fps)
            voi[t] = abs(di[t] - di[t - 1]) / dt
        all_vrel.append(voi[1:])    # exclude frame 0
    v_th = _p10(np.concatenate(all_vrel)) if all_vrel else 1e-6

    # ── R_ref (P90 of hand spread) ────────────────────────────────────────────
    R_vals = []
    for rec in records:
        if rec.hand is not None:
            R_vals.append(rec.hand.hand_spread())
    R_ref = _p90(np.array(R_vals)) if R_vals else 1.0

    # ── sigma_d (P90 of min fingertip-object distances) ───────────────────────
    all_d_min = []
    for label, poi in obj_pos.items():
        for t, rec in enumerate(records):
            if rec.hand is None:
                continue
            fingertips = rec.hand.fingertips
            diffs      = fingertips - poi[t][np.newaxis, :]
            d_min      = float(np.min(np.linalg.norm(diffs, axis=1)))
            all_d_min.append(d_min)
    sigma_d = _p90(np.array(all_d_min)) if all_d_min else 1.0

    return dict(d_max=d_max, vh_max=vh_max, v_th=v_th,
                R_ref=R_ref, sigma_d=sigma_d)


# ─────────────────────────────────────────────────────────────────────────────
# Per-object cost evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_costs_for_object(
    label:   str,
    ph:      np.ndarray,
    vh:      np.ndarray,
    poi:     np.ndarray,
    records: List[FrameRecord],
    fps:     float,
    norms:   dict,
) -> Dict[str, np.ndarray]:
    """
    Compute all seven cost terms and J_i(t) for a single object i.

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
    phi_comp   = hand_compactness_cost(records, poi, R_ref, sigma_d)
    phi_enc    = enclosure_cost(records, poi)
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
) -> None:
    """
    Write cost time-histories for every object to a formatted .txt file.

    File structure
    --------------
    A header section followed by one DATA BLOCK per object.
    Inside each block, every row corresponds to one video frame:

      frame_idx  timestamp_ms  phi_d  phi_v  phi_dir  phi_obj  phi_comp  phi_enc  phi_couple  J

    Column widths are fixed for easy parsing with numpy.loadtxt or pandas.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    TERM_NAMES = ["phi_d", "phi_v", "phi_dir", "phi_obj",
                  "phi_comp", "phi_enc", "phi_couple", "J"]
    COL_W      = 14   # fixed column width

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
        fh.write(f"# Tracked objects : {', '.join(obj_labels)}\n")
        fh.write("#\n")
        fh.write("# Cost terms (paper equations):\n")
        fh.write("#   phi_d      – distance cost              \n")
        fh.write("#   phi_v      – hand velocity cost         \n")
        fh.write("#   phi_dir    – hand direction cost        \n")
        fh.write("#   phi_obj    – object velocity cost       \n")
        fh.write("#   phi_comp   – hand compactness cost      \n")
        fh.write("#   phi_enc    – enclosure cost             \n")
        fh.write("#   phi_couple – coupling term              \n")
        fh.write("#   J          – total cost                 \n")
        fh.write("# ============================================================\n\n")

        for label in obj_labels:
            obj_costs = costs[label]

            # ── Object block header ───────────────────────────────────────────
            fh.write(f"# {'=' * 58}\n")
            fh.write(f"# OBJECT: {label}\n")
            fh.write(f"# {'=' * 58}\n")

            # ── Column header ─────────────────────────────────────────────────
            col_names = ["frame_idx", "timestamp_ms"] + TERM_NAMES
            header_parts = []
            for name in col_names:
                header_parts.append(f"{name:>{COL_W}}")
            fh.write("#" + "".join(header_parts) + "\n")

            # ── Data rows ─────────────────────────────────────────────────────
            T = len(records)
            for t in range(T):
                row_parts = [
                    f"{records[t].frame_idx:{COL_W}d}",
                    f"{records[t].timestamp_ms:{COL_W}.3f}",
                ]
                for term in TERM_NAMES:
                    row_parts.append(_fmt(float(obj_costs[term][t])))
                fh.write(" ".join(row_parts) + "\n")

            fh.write("\n")   # blank line between object blocks

    print(f"[Writer] Cost results saved to '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_costs(
    tracking_path: str,
    output_path:   str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Full cost-function evaluation pipeline.
    Returns
    -------
    costs : dict[label, dict[term_name, np.ndarray]]
    """
    # ── 1. Load tracking data ─────────────────────────────────────────────────
    records, fps = parse_tracking_file(tracking_path)
    if not records:
        raise RuntimeError("No frame records found in the tracking file.")

    obj_labels = collect_object_labels(records)
    if not obj_labels:
        raise RuntimeError("No objects found in the tracking file.")

    print(f"[Cost]   Objects detected: {obj_labels}")

    # ── 2. Kinematic arrays ────────────────────────────────────────────────────
    ph      = compute_hand_positions(records)
    vh      = compute_hand_velocities(ph, records, fps)
    obj_pos = compute_object_positions(records, obj_labels)

    # ── 3. Global normalisation constants ─────────────────────────────────────
    norms = compute_global_normalisation(ph, vh, obj_pos, records, fps)
    print("[Cost]   Normalisation constants:")
    for k, v in norms.items():
        print(f"           {k:10s} = {v:.4f}")

    # ── 4. Cost evaluation per object ──────────────────────────────────────────
    all_costs: Dict[str, Dict[str, np.ndarray]] = {}
    for label in obj_labels:
        print(f"[Cost]   Computing costs for '{label}' ...")
        all_costs[label] = evaluate_costs_for_object(
            label   = label,
            ph      = ph,
            vh      = vh,
            poi     = obj_pos[label],
            records = records,
            fps     = fps,
            norms   = norms,
        )

    # ── 5. Save to file ────────────────────────────────────────────────────────
    save_cost_results(output_path, records, obj_labels, all_costs, fps)

    return all_costs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    costs = compute_all_costs(TRACKING_RESULTS_PATH, COST_OUTPUT_PATH)

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n[Summary]  Mean cost values across all frames:")
    print(f"  {'Object':<25s}  {'mean J':>10s}  {'min J':>10s}  {'max J':>10s}")
    print("  " + "-" * 60)
    for label, obj_costs in costs.items():
        J = obj_costs["J"]
        print(f"  {label:<25s}  {np.mean(J):10.4f}  {np.min(J):10.4f}  {np.max(J):10.4f}")

    print("\n[DONE]")