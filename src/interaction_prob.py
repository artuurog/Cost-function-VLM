"""
interaction_probability.py

Input
-----
  Cost results .txt file produced by cost_function.py.

Output
------
  interaction_probability.txt — P_i(t) time-history for every object
                                 + keyframe table at the end of the file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter, find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# User settings
# ─────────────────────────────────────────────────────────────────────────────

COST_FILE_PATH   = "results/cost_function.txt"
OUTPUT_PATH      = "results/interaction_probability.txt"

# Savitzky-Golay filter parameters
# window_length must be odd and > polyorder; increase to smooth more aggressively
SG_WINDOW_LENGTH: int = 11
SG_POLYORDER:     int = 3

# Minimum number of frames between two consecutive keyframes (avoids duplicates)
MIN_PEAK_DISTANCE: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

class Keyframe(NamedTuple):
    """A selected keyframe and its associated dominant object."""
    frame_idx:       int     # original video frame index
    timestamp_ms:    float   # timestamp in milliseconds
    dominant_object: str     # label of the object with highest P at this frame
    probability:     float   # P_i*(t*) of the dominant object


# ─────────────────────────────────────────────────────────────────────────────
# Cost file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_cost_file(
    path: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Read the cost_function.txt file produced by cost_function.py.

    Returns
    -------
    obj_labels   : list of object label strings, length N
    frame_indices: integer array, shape (T,)
    timestamps   : float array of timestamps in ms, shape (T,)
    J            : dict[label -> 1-D float array of shape (T,)]
                   total cost time-history per object
    """
    obj_labels:   List[str]              = []
    frame_idx_by_obj: Dict[str, List[int]]   = {}
    ts_by_obj:        Dict[str, List[float]] = {}
    J_by_obj:         Dict[str, List[float]] = {}

    current_label: Optional[str] = None

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            # ── Object block header ───────────────────────────────────────────
            # Looks like:  # OBJECT: red_block
            m = re.match(r"^#\s*OBJECT:\s*(.+)$", line)
            if m:
                current_label = m.group(1).strip()
                if current_label not in obj_labels:
                    obj_labels.append(current_label)
                    frame_idx_by_obj[current_label] = []
                    ts_by_obj[current_label]        = []
                    J_by_obj[current_label]         = []
                continue

            # Skip all other comment / column-header lines
            if line.startswith("#"):
                continue

            # ── Data row ──────────────────────────────────────────────────────
            # Columns: frame_idx  timestamp_ms  phi_d  phi_v  phi_dir  phi_obj
            #          phi_comp  phi_enc  phi_couple  J
            if current_label is None:
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            try:
                fidx = int(tokens[0])
                ts   = float(tokens[1])
                j    = float(tokens[9])        # column index 9 = J
            except ValueError:
                continue

            frame_idx_by_obj[current_label].append(fidx)
            ts_by_obj[current_label].append(ts)
            J_by_obj[current_label].append(j)

    if not obj_labels:
        raise RuntimeError(f"No object blocks found in '{path}'.")

    # Verify all objects have the same number of frames
    lengths = {lb: len(J_by_obj[lb]) for lb in obj_labels}
    T = list(lengths.values())[0]
    for lb, n in lengths.items():
        if n != T:
            raise RuntimeError(
                f"Object '{lb}' has {n} frames but expected {T}. "
                "Cost file may be corrupted."
            )

    frame_indices = np.array(frame_idx_by_obj[obj_labels[0]], dtype=np.int64)
    timestamps    = np.array(ts_by_obj[obj_labels[0]],        dtype=np.float64)
    J = {lb: np.array(J_by_obj[lb], dtype=np.float64) for lb in obj_labels}

    print(f"[Parser] Loaded costs for {len(obj_labels)} object(s), "
          f"{T} frames each.")
    print(f"[Parser] Objects: {obj_labels}")

    return obj_labels, frame_indices, timestamps, J


# ─────────────────────────────────────────────────────────────────────────────
# Temperature-scaled softmax
# ─────────────────────────────────────────────────────────────────────────────

def compute_temperature(J_i: np.ndarray, N: int) -> float:
    """

    Parameters
    ----------
    J_i : cost time-series for object i, shape (T,)
    N   : total number of tracked objects in the scene

    Returns
    -------
    tau_i : temperature scalar (> 0)
    """
    sigma = float(np.std(J_i, ddof=0))
    tau   = sigma / np.sqrt(max(N, 2) / 2.0)
    # Guard against degenerate (constant) cost signals
    return max(tau, 1e-9)


def softmax_probability(
    J:          Dict[str, np.ndarray],
    obj_labels: List[str],
) -> Dict[str, np.ndarray]:
    """
    Convert costs J_i(t) to interaction probabilities P_i(t) via
    temperature-scaled softmax

    Parameters
    ----------
    J          : dict[label -> cost array shape (T,)]
    obj_labels : ordered list of object labels

    Returns
    -------
    P : dict[label -> probability array shape (T,)]  each in [0, 1],
        and sum over objects at every t equals 1.
    """
    N = len(obj_labels)
    # Pre-compute per-object temperatures
    tau = {lb: compute_temperature(J[lb], N) for lb in obj_labels}

    # Shape (N, T)
    log_unnorm = np.stack(
        [-J[lb] / tau[lb] for lb in obj_labels], axis=0
    )  # (N, T)

    # Numerically stable softmax: subtract row-wise max
    log_unnorm -= log_unnorm.max(axis=0, keepdims=True)
    exp_vals    = np.exp(log_unnorm)                   # (N, T)
    denom       = exp_vals.sum(axis=0, keepdims=True)  # (1, T)
    P_matrix    = exp_vals / denom                     # (N, T)

    P = {lb: P_matrix[i] for i, lb in enumerate(obj_labels)}

    print("[Softmax] Computed P_i(t) for all objects.")
    for lb in obj_labels:
        print(f"          {lb:<30s}  tau={tau[lb]:.4f}  "
              f"mean P={np.mean(P[lb]):.4f}  max P={np.max(P[lb]):.4f}")

    return P


# ─────────────────────────────────────────────────────────────────────────────
# Savitzky-Golay smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_probability(
    P:             np.ndarray,
    window_length: int = SG_WINDOW_LENGTH,
    polyorder:     int = SG_POLYORDER,
) -> np.ndarray:
    """

    Parameters
    ----------
    P             : raw probability array, shape (T,)
    window_length : SG filter window (must be odd, > polyorder)
    polyorder     : SG polynomial order

    Returns
    -------
    P_smooth : smoothed probability array, shape (T,)
               clipped to [0, 1] to correct for filter edge artefacts.
    """
    T = len(P)
    # Ensure window_length is valid
    wl = min(window_length, T)
    if wl % 2 == 0:
        wl -= 1                     # must be odd
    wl = max(wl, polyorder + 2)     # must exceed polynomial order

    P_smooth = savgol_filter(P, window_length=wl, polyorder=polyorder)
    return np.clip(P_smooth, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection + statistical threshold
# ─────────────────────────────────────────────────────────────────────────────

def detect_interaction_peaks(
    P_smooth:          np.ndarray,
    min_peak_distance: int = MIN_PEAK_DISTANCE,
) -> Tuple[np.ndarray, float]:
    """

    Parameters
    ----------
    P_smooth          : smoothed probability array, shape (T,)
    min_peak_distance : minimum separation between two consecutive peaks
                        (frames); maps to the `distance` parameter of
                        scipy.signal.find_peaks.

    Returns
    -------
    peak_indices : integer array of retained peak positions in [0, T)
    threshold    : the P90 value used for filtering
    """
    #raw peak detection
    raw_peaks, _ = find_peaks(
        P_smooth,
        distance=min_peak_distance,
    )

    if len(raw_peaks) == 0:
        return np.array([], dtype=np.int64), 0.0

    # statistical threshold: P90 of the smoothed signal
    threshold = float(np.percentile(P_smooth, 90))

    # Keep only peaks above the threshold
    filtered = raw_peaks[P_smooth[raw_peaks] > threshold]

    return filtered.astype(np.int64), threshold


# ─────────────────────────────────────────────────────────────────────────────
# Dominant object selection and keyframe aggregation
# ─────────────────────────────────────────────────────────────────────────────

def select_keyframes(
    P:            Dict[str, np.ndarray],
    P_smooth:     Dict[str, np.ndarray],
    obj_labels:   List[str],
    frame_indices: np.ndarray,
    timestamps:   np.ndarray,
    peak_indices_per_obj: Dict[str, np.ndarray],
) -> List[Keyframe]:
    """

    Parameters
    ----------
    P                    : raw probability dict[label -> (T,)]
    P_smooth             : smoothed probability dict[label -> (T,)]
    obj_labels           : ordered list of object labels
    frame_indices        : original video frame indices, shape (T,)
    timestamps           : timestamps in ms, shape (T,)
    peak_indices_per_obj : dict[label -> peak index array (indices into T)]

    Returns
    -------
    keyframes : chronologically sorted list of Keyframe namedtuples
    """
    # Collect all candidate time positions across all objects
    candidate_set: set = set()
    for peaks in peak_indices_per_obj.values():
        candidate_set.update(peaks.tolist())

    # Build interaction probability matrix  (N, T)
    P_smooth_matrix = np.stack(
        [P_smooth[lb] for lb in obj_labels], axis=0
    )  # (N, T)

    keyframes: List[Keyframe] = []
    for t_idx in sorted(candidate_set):
        # Dominant object = argmax over smoothed P at this frame
        i_star      = int(np.argmax(P_smooth_matrix[:, t_idx]))
        dominant_lb = obj_labels[i_star]
        prob_val    = float(P_smooth_matrix[i_star, t_idx])

        keyframes.append(Keyframe(
            frame_idx       = int(frame_indices[t_idx]),
            timestamp_ms    = float(timestamps[t_idx]),
            dominant_object = dominant_lb,
            probability     = prob_val,
        ))

    # Sort chronologically
    keyframes.sort(key=lambda kf: kf.frame_idx)

    return keyframes


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def save_probability_results(
    output_path:  str,
    obj_labels:   List[str],
    frame_indices: np.ndarray,
    timestamps:   np.ndarray,
    P_raw:        Dict[str, np.ndarray],
    P_smooth:     Dict[str, np.ndarray],
    thresholds:   Dict[str, float],
    keyframes:    List[Keyframe],
) -> None:
    """
   
    File structure
    --------------
    [Header]
    [Per-object probability block]   — one block per tracked object
        Columns: frame_idx  timestamp_ms  P_raw  P_smooth
    [Keyframe table]
        Columns: keyframe_rank  frame_idx  timestamp_ms  dominant_object  probability
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    T   = len(frame_indices)
    COL = 16   # fixed column width

    def _f(x: float) -> str:
        return f"{x:{COL}.6f}"

    with open(output_path, "w") as fh:

        # ── File header ───────────────────────────────────────────────────────
        fh.write("# ============================================================\n")
        fh.write("# Hand-Object Interaction Probability  —  time history\n")
        fh.write("#\n")
        fh.write(f"# Cost source     : {COST_FILE_PATH}\n")
        fh.write(f"# Total frames    : {T}\n")
        fh.write(f"# Tracked objects : {', '.join(obj_labels)}\n")
        fh.write(f"# Keyframes found : {len(keyframes)}\n")
        fh.write("#\n")
        fh.write("# Method:\n")
        fh.write("#   1. Temperature-scaled softmax  P_i(t)\n")
        fh.write("#   2. Savitzky-Golay smoothing     P~_i(t)\n")
        fh.write("#   3. Local maximum detection\n")
        fh.write("#   4. P90 statistical threshold\n")
        fh.write("#   5. Dominant-object selection    i*(t*)\n")
        fh.write("# ============================================================\n\n")

        # ── Per-object probability blocks ─────────────────────────────────────
        for lb in obj_labels:
            fh.write(f"# {'=' * 58}\n")
            fh.write(f"# OBJECT: {lb}\n")
            fh.write(f"#   P90 threshold used for peak filtering: "
                     f"{thresholds[lb]:.6f}\n")
            fh.write(f"# {'=' * 58}\n")

            # Column header
            cols = ["frame_idx", "timestamp_ms", "P_raw", "P_smooth"]
            fh.write("#" + "".join(f"{c:>{COL}}" for c in cols) + "\n")

            for t in range(T):
                row = [
                    f"{frame_indices[t]:{COL}d}",
                    f"{timestamps[t]:{COL}.3f}",
                    _f(P_raw[lb][t]),
                    _f(P_smooth[lb][t]),
                ]
                fh.write(" ".join(row) + "\n")

            fh.write("\n")

        # ── Keyframe table ────────────────────────────────────────────────────
        fh.write("# ============================================================\n")
        fh.write("# KEYFRAME TABLE\n")
        fh.write("#   Each row = one extracted keyframe (local probability max)\n")
        fh.write("#   dominant_object = object i* with highest P~_i at that frame\n")
        fh.write("# ============================================================\n")

        kf_cols = ["rank", "frame_idx", "timestamp_ms",
                   "dominant_object", "probability"]
        fh.write("#" + "".join(f"{c:>{COL}}" for c in kf_cols) + "\n")

        for rank, kf in enumerate(keyframes, start=1):
            row = [
                f"{rank:{COL}d}",
                f"{kf.frame_idx:{COL}d}",
                f"{kf.timestamp_ms:{COL}.3f}",
                f"{kf.dominant_object:>{COL}s}",
                _f(kf.probability),
            ]
            fh.write(" ".join(row) + "\n")

    print(f"[Writer] Probability results saved to '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_probability(
    cost_path:   str = COST_FILE_PATH,
    output_path: str = OUTPUT_PATH,
) -> Tuple[Dict[str, np.ndarray], List[Keyframe]]:
    """
    Parameters
    ----------
    cost_path   : path to the cost_function.txt file
    output_path : path for the output interaction_probability.txt file

    Returns
    -------
    P_smooth  : dict[label -> smoothed probability array shape (T,)]
    keyframes : chronologically ordered list of Keyframe namedtuples
    """

    # ── Parse costs ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Parse cost file")
    print("=" * 60)
    obj_labels, frame_indices, timestamps, J = parse_cost_file(cost_path)

    # ── Temperature-scaled softmax ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Temperature-scaled softmax  [Eq. 13-14]")
    print("=" * 60)
    P_raw = softmax_probability(J, obj_labels)

    # ── Savitzky-Golay smoothing ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Savitzky-Golay smoothing")
    print("=" * 60)
    P_smooth: Dict[str, np.ndarray] = {}
    for lb in obj_labels:
        P_smooth[lb] = smooth_probability(P_raw[lb])
        print(f"  {lb:<30s}  window={SG_WINDOW_LENGTH}  poly={SG_POLYORDER}")

    # ── Peak detection + statistical threshold ─────────────────────
    print("\n" + "=" * 60)
    print("  Peak detection + P90 threshold")
    print("=" * 60)
    peak_indices_per_obj: Dict[str, np.ndarray] = {}
    thresholds:           Dict[str, float]       = {}

    for lb in obj_labels:
        peaks, thr = detect_interaction_peaks(P_smooth[lb])
        peak_indices_per_obj[lb] = peaks
        thresholds[lb]           = thr
        print(f"  {lb:<30s}  threshold={thr:.4f}  peaks={len(peaks)}"
              + (f"  @ frames {frame_indices[peaks].tolist()}" if len(peaks) else ""))

    # ── Dominant object selection → keyframes ─────────────────────────
    print("\n" + "=" * 60)
    print("  Dominant object selection  →  keyframes")
    print("=" * 60)
    keyframes = select_keyframes(
        P_raw, P_smooth, obj_labels,
        frame_indices, timestamps,
        peak_indices_per_obj,
    )

    # ── Save results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Save results")
    print("=" * 60)
    save_probability_results(
        output_path   = output_path,
        obj_labels    = obj_labels,
        frame_indices = frame_indices,
        timestamps    = timestamps,
        P_raw         = P_raw,
        P_smooth      = P_smooth,
        thresholds    = thresholds,
        keyframes     = keyframes,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEYFRAME SUMMARY")
    print("=" * 60)
    print(f"  {'Rank':>5}  {'Frame':>7}  {'Time (ms)':>12}  "
          f"{'Dominant object':<30}  {'P̃':>8}")
    print("  " + "-" * 68)
    for rank, kf in enumerate(keyframes, start=1):
        print(f"  {rank:>5}  {kf.frame_idx:>7}  {kf.timestamp_ms:>12.1f}  "
              f"{kf.dominant_object:<30}  {kf.probability:>8.4f}")

    print(f"\n  Total keyframes extracted: {len(keyframes)}")
    print("=" * 60)

    return P_smooth, keyframes


# ─────────────────────────────────────────────────────────────────────────────
# Convenience accessor
# ─────────────────────────────────────────────────────────────────────────────

def get_keyframe_indices(keyframes: List[Keyframe]) -> List[int]:
    """

    Parameters
    ----------
    keyframes : output of compute_interaction_probability()

    Returns
    -------
    list of integer frame indices
    """
    return [kf.frame_idx for kf in keyframes]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    P_smooth, keyframes = compute_interaction_probability(
        cost_path   = COST_FILE_PATH,
        output_path = OUTPUT_PATH,
    )

    # Expose the final keyframe indices as a plain list
    kf_indices = get_keyframe_indices(keyframes)
    print(f"\n[DONE]  Keyframe indices: {kf_indices}")