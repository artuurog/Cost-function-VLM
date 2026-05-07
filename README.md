# VLM Keyframe Extraction via Hand-Object Interaction Cost Function

A model-based keyframe extraction pipeline for robotic task learning from human video demonstrations. Given an RGB video of a human manipulation task, this codebase identifies the most informative frames — those corresponding to meaningful hand-object interactions — and delivers them to a Vision-Language Model (VLM) for task plan generation.

This repository implements the methodology described in the paper "Keyframe Extraction from Human Demonstrations via Hand-Object Interaction Cost Function for VLM-Based Task Learning"*

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1 — Object Discovery (`track_objects.py`)](#step-1--object-discovery-track_objectspy)
  - [Step 2 — Hand and Object Tracking (`track_combined.py`)](#step-2--hand-and-object-tracking-track_combinedpy)
  - [Step 3 — Cost Function Evaluation (`cost_function.py`)](#step-3--cost-function-evaluation-cost_functionpy)
  - [Step 4 — Interaction Probability and Keyframe Selection (`interaction_prob.py`)](#step-4--interaction-probability-and-keyframe-selection-interaction_probpy)
  - [Step 5 — Keyframe Extraction (`keyframes.py`)](#step-5--keyframe-extraction-keyframespy)
- [Data Formats](#data-formats)
- [Module Reference](#module-reference)
- [Mathematical Background](#mathematical-background)
- [Results and Output Files](#results-and-output-files)

---

## Overview

Standard approaches to learning manipulation tasks from video either send the entire video to a VLM (which quickly saturates the context window) or sample frames at a fixed frequency (which may miss critical interaction moments). This project implements an **adaptive keyframe sampling** strategy grounded in a physics-inspired, multi-term cost function that measures the quality of hand-object interaction at every frame.

The pipeline proceeds as follows:

1. A VLM identifies all objects in the first frame of the video.
2. An open-vocabulary object detector (GroundingDINO) and a hand tracker (MediaPipe) localize all entities frame by frame.
3. A seven-term cost function `J_i(t)` is evaluated for every hand-object pair `(hand, object_i)` at every frame `t`.
4. A temperature-scaled softmax converts the costs into interaction probabilities `P_i(t)`.
5. The frames corresponding to local probability maxima are extracted as keyframes, each annotated with the dominant interacting object.
6. The keyframes are fed sequentially to a VLM for task understanding and PDDL plan generation.

The result is a significant **frame reduction** (typically >90% of frames are discarded) while retaining the semantically critical moments of the demonstration.

## Repository Structure

```
.
├── track_objects.py          # Step 1: VLM-based object discovery from first frame
├── track_combined.py         # Step 2: Full frame-by-frame hand + object tracking
├── utils.py                  # Shared data structures and tracking utilities
├── cost_function.py          # Step 3: Seven-term hand-object interaction cost
├── interaction_prob.py       # Step 4: Softmax → probabilities → keyframe selection
├── keyframes.py              # Step 5: Extract keyframe images / summary video
├── results/                  # Auto-created output directory
│   ├── tracking_results.txt
│   ├── cost_function.txt
│   ├── interaction_probability.txt
│   └── keyframes/
│       ├── kf_001_frame0042_cup.jpg
│       └── ...
└── VLM_keyframe_cost_function_PREPRINT.pdf   # Reference paper
```

---

## Dependencies

Install all Python dependencies with:

```bash
pip install opencv-python mediapipe openai pillow numpy scipy torch torchvision transformers
```

| Package | Purpose |
|---|---|
| `opencv-python` | Video I/O, frame annotation |
| `mediapipe` | 21-keypoint hand detection and tracking |
| `openai` | VLM API calls (via HuggingFace router or OpenAI) |
| `pillow` | Image encoding for API payloads |
| `numpy` | All numerical array operations |
| `scipy` | Savitzky-Golay smoothing, peak finding |
| `torch` / `transformers` | GroundingDINO object detection |

**External model access:**

- A **HuggingFace API token** (`HF_TOKEN`) is required for VLM calls in `track_objects.py`. The default model is `allenai/Molmo2-8B`, accessed through `https://router.huggingface.co/v1`.
- An **OpenAI-compatible API key** (`OPENAI_API_KEY`) is required in `track_combined.py` for the object-listing VLM step.

---

## Configuration

Each script contains a clearly delimited `User settings` block at the top. Edit these constants before running:

**`track_objects.py`**
```python
VIDEO_PATH = "path/to/your/video.mp4"
HF_TOKEN   = "hf_..."
```

**`track_combined.py`**
```python
VIDEO_PATH     = "path/to/your/video.mp4"
OUTPUT_PATH    = "results/tracking_results.txt"
OPENAI_API_KEY = "sk-..."
DINO_THRESHOLD = 0.30       # GroundingDINO confidence threshold
REDETECT_EVERY = 30         # Re-run detection every N frames (0 = disabled)
DISPLAY        = True       # Show live annotated video
SKIP_FRAMES    = 0          # Subsample rate (0 = process every frame)
```

**`cost_function.py`**
```python
TRACKING_RESULTS_PATH = "results/tracking_results.txt"
COST_OUTPUT_PATH      = "results/cost_function.txt"
```

**`interaction_prob.py`**
```python
COST_FILE_PATH    = "results/cost_function.txt"
OUTPUT_PATH       = "results/interaction_probability.txt"
SG_WINDOW_LENGTH  = 11      # Savitzky-Golay window (must be odd)
SG_POLYORDER      = 3       # Savitzky-Golay polynomial order
MIN_PEAK_DISTANCE = 5       # Minimum frames between two peaks
```

**`keyframes.py`**
```python
VIDEO_PATH       = "path/to/your/video.mp4"
PROBABILITY_FILE = "results/interaction_probability.txt"
OUTPUT_DIR       = "results/keyframes"
EXTRACTION_MODE  = "files"  # "files" → one JPEG per keyframe
                             # "video" → summary .mp4 clip
JPEG_QUALITY     = 95
SUMMARY_VIDEO_FPS = 2.0
RUN_PIPELINE_IF_NEEDED = False   # Auto-run interaction_prob.py if needed
```

---

## Pipeline Walkthrough

### Step 1 — Object Discovery (`track_objects.py`)

This optional standalone script sends the **first frame** of the video to a VLM and asks it to identify all objects on the workspace table, assigning each a unique text label and a pixel coordinate.

**What it does:**
1. Opens the video with OpenCV and reads frame 0.
2. Encodes the frame as a base64 JPEG.
3. Sends it to `allenai/Molmo2-8B` via the HuggingFace inference router with a structured prompt.
4. Parses the response with a regex expecting the format `OBJECT_NAME: (u, v)`.
5. Displays an annotated frame showing each detected object and its label.

**Output:** A printed list of object labels and pixel positions. These labels are used as prompts for GroundingDINO in the next step.

**Run:**
```bash
python track_objects.py
```

---

### Step 2 — Hand and Object Tracking (`track_combined.py`)

The main tracking loop. For each frame of the video, the pipeline:

1. **Object initialization (frame 0 only):** Calls `list_objects_vlm()` (from `utils.py`) to get object labels from the VLM, then runs `detect_objects_dino()` to get bounding boxes and segmentation masks.
2. **Per-frame tracking:** Uses `MultiObjectTracker` (wrapping GroundingDINO re-detection on a configurable interval) for objects and `HandTracker` (wrapping MediaPipe Hands) for the hand.
3. **Kinematics:** Computes `compute_relative_distance()` and `compute_relative_velocity()` between the hand and each object.
4. **Annotation:** Optionally draws hand keypoints, bounding boxes, and object centroids on a live display window.
5. **Output:** Calls `save_results_txt()` to write all tracking data to `results/tracking_results.txt`.

The shared data structures (`HandData`, `ObjectData`, `FrameResult`) are defined in `utils.py` and used throughout this step and the next.

**Run:**
```bash
python track_combined.py
```

---

### Step 3 — Cost Function Evaluation (`cost_function.py`)

Reads `tracking_results.txt` and computes a **seven-term interaction cost** `J_i(t)` for every object `i` at every frame `t`. Lower cost = higher likelihood of meaningful interaction.

**The seven cost terms** (see [Mathematical Background](#mathematical-background) for full equations):

| Term | Variable | Description |
|---|---|---|
| Distance cost | `phi_d` | Normalized Euclidean distance between hand centroid and object centroid |
| Hand velocity cost | `phi_v` | Normalized magnitude of hand speed |
| Direction cost | `phi_dir` | Penalizes hand motion not directed toward the object |
| Object velocity cost | `phi_obj` | Penalizes large relative hand-object velocity (active manipulation) |
| Hand compactness cost | `phi_comp` | Measures finger closure (grip posture) near the object |
| Enclosure cost | `phi_enc` | Fraction of object bounding box corners enclosed by the hand keypoint convex hull |
| Coupling term | `phi_couple` | Multiplicative coupling between distance and velocity signals |

The **total cost** is:
```
J_i(t) = phi_d + phi_v + phi_dir + phi_obj + phi_comp + phi_enc + phi_couple
```

All terms are normalized using **robust statistics** (P90 and P10 percentiles computed globally across all objects and frames) to avoid sensitivity to tracking outliers.

**Key functions:**

- `parse_tracking_file(path)` — Loads `tracking_results.txt` into a list of `FrameRecord` objects.
- `compute_hand_positions(records)` — Builds `p_h(t)` with NaN-filling for occluded frames.
- `compute_global_normalisation(...)` — Computes `d_max`, `vh_max`, `v_th`, `R_ref`, `sigma_d`.
- `evaluate_costs_for_object(...)` — Returns all seven cost arrays for one object.
- `compute_all_costs(tracking_path, output_path)` — Top-level function; runs everything and writes output.

**Run:**
```bash
python cost_function.py
```

---

### Step 4 — Interaction Probability and Keyframe Selection (`interaction_prob.py`)

Converts the raw costs into a normalized probability distribution and identifies the frames most likely to contain meaningful interactions.

**Processing pipeline:**

1. **Temperature-scaled softmax** (Eq. 13–14 of the paper): For each frame `t`, the costs of all `N` objects are converted to probabilities:
   ```
   P_i(t) = exp(-J_i(t) / tau_i) / sum_k exp(-J_k(t) / tau_k)
   ```
   where `tau_i = sigma_Ji / (sqrt(N) / 2)` is an adaptive temperature computed from the standard deviation of `J_i` over the full video.

2. **Savitzky-Golay smoothing:** The raw probability signal `P_i(t)` is smoothed with a polynomial filter (default: window=11, order=3) to suppress high-frequency tracking noise while preserving peak structure.

3. **Peak detection:** `scipy.signal.find_peaks` detects local maxima in the smoothed signal with a minimum separation of `MIN_PEAK_DISTANCE` frames.

4. **P90 statistical threshold:** Only peaks whose smoothed probability exceeds the 90th percentile of the smoothed signal are retained.

5. **Dominant object selection:** For each retained peak frame `t*`, the object `i*` with the highest smoothed probability is selected as the dominant interacting object.

**Key functions:**

- `parse_cost_file(path)` — Reads `cost_function.txt` into per-object cost arrays.
- `compute_softmax_probabilities(J, obj_labels)` — Applies the temperature-scaled softmax.
- `smooth_and_find_peaks(P_smooth, ...)` — Applies SG filter and returns filtered peak indices.
- `select_keyframes(...)` — Aggregates peaks across all objects and selects dominant object per keyframe.
- `compute_interaction_probability(cost_path, output_path)` — Top-level function.

**Run:**
```bash
python interaction_prob.py
```

---

### Step 5 — Keyframe Extraction (`keyframes.py`)

Reads the keyframe table from `interaction_probability.txt`, seeks to the corresponding frames in the original video, and saves them.

**Two extraction modes:**
- `"files"` — Saves one annotated JPEG per keyframe, named `kf_NNN_frameXXXXX_<object>.jpg`. Each image is annotated with the frame index, timestamp, dominant object label, and interaction probability.
- `"video"` — Writes all keyframes sequentially into a summary `.mp4` clip at a configurable FPS.

**Key functions:**

- `parse_keyframe_table(path)` — Parses the `KEYFRAME TABLE` section of `interaction_probability.txt`.
- `extract_keyframe_images(video_path, keyframes, output_dir, ...)` — Saves one JPEG per keyframe.
- `extract_keyframe_video(video_path, keyframes, output_path, ...)` — Writes the summary video.

**Run:**
```bash
python keyframes.py
```

---

## Data Formats

### `results/tracking_results.txt`

One block per frame. Each block starts with a `FRAME` line followed by optional `HAND` and `OBJECT` lines:

```
# FPS: 30.00
FRAME 0 0.000
HAND Right  <cx> <cy>  <kp0x> <kp0y> ... <kp20x> <kp20y>  <bx1> <by1> <bx2> <by2>
OBJECT red_block  <cx> <cy>  <bx1> <by1> <bx2> <by2>
OBJECT blue_cup   <cx> <cy>  <bx1> <by1> <bx2> <by2>
FRAME 1 33.333
...
```

The `HAND` line encodes: 2 centroid floats + 42 keypoint floats (21 × 2) + 4 bbox ints = 48 values total.

### `results/cost_function.txt`

One data block per object. Each block has a column header comment followed by fixed-width rows:

```
# OBJECT: red_block
#    frame_idx  timestamp_ms    phi_d    phi_v  phi_dir  phi_obj  phi_comp  phi_enc  phi_couple          J
         0         0.000   0.312451   0.089234  ...
         1        33.333   0.298771   0.091122  ...
```

### `results/interaction_probability.txt`

Contains per-object probability time histories followed by the final keyframe table:

```
# OBJECT: red_block
#   P90 threshold used for peak filtering: 0.412300
#    frame_idx  timestamp_ms    P_raw   P_smooth
          0        0.000    0.5102    0.4987
          ...

# KEYFRAME TABLE
#     rank  frame_idx  timestamp_ms        dominant_object  probability
         1         42       1400.0            red_block      0.7823
         2        107       3566.7                blue_cup   0.6541
```

---

## Module Reference

### `utils.py`

Central utility module. Contains:

- **Data structures:** `HandData`, `ObjectData`, `FrameResult` — dataclasses used by the tracker and cost function.
- **Video I/O:** `extract_first_frame(video_path)` — returns the first BGR frame and video metadata.
- **VLM interface:** `list_objects_vlm(frame, api_key, model)` — encodes a frame and calls the VLM to list workspace objects.
- **Detection:** `detect_objects_dino(frame, labels, threshold)` — runs GroundingDINO to get bounding boxes; returns `ObjectData` instances.
- **Tracking classes:**
  - `MultiObjectTracker` — manages per-frame object re-detection and tracks centroids using the GroundingDINO bounding boxes.
  - `HandTracker` — wraps MediaPipe Hands; fuses 21-keypoint output with bounding-box centroid for robustness to partial occlusion.
- **Kinematics:** `compute_relative_distance()`, `compute_relative_velocity()`.
- **Output:** `save_results_txt(path, results, fps)` — writes `tracking_results.txt`.

### `cost_function.py`

- `parse_tracking_file(path)` → `(List[FrameRecord], float fps)`
- `compute_hand_positions(records)` → `np.ndarray (T, 2)`
- `compute_hand_velocities(ph, records, fps)` → `np.ndarray (T, 2)`
- `compute_object_positions(records, obj_labels)` → `Dict[str, np.ndarray (T, 2)]`
- `compute_global_normalisation(...)` → `dict` with keys `d_max`, `vh_max`, `v_th`, `R_ref`, `sigma_d`
- `distance_cost(ph, poi, d_max)` → `np.ndarray (T,)`
- `hand_velocity_cost(vh, vh_max)` → `np.ndarray (T,)`
- `hand_direction_cost(vh, ph, poi)` → `np.ndarray (T,)`
- `object_velocity_cost(ph, poi, records, fps, v_th)` → `np.ndarray (T,)`
- `hand_compactness_cost(records, poi, R_ref, sigma_d)` → `np.ndarray (T,)`
- `enclosure_cost(records, poi)` → `np.ndarray (T,)`
- `coupling_term(phi_d, phi_v)` → `np.ndarray (T,)`
- `evaluate_costs_for_object(...)` → `Dict[str, np.ndarray]`
- `compute_all_costs(tracking_path, output_path)` → `Dict[str, Dict[str, np.ndarray]]`

### `interaction_prob.py`

- `parse_cost_file(path)` → `(obj_labels, frame_indices, timestamps, J)`
- `compute_softmax_probabilities(J, obj_labels)` → `(P_raw, temperatures)`
- `smooth_and_find_peaks(P_smooth, min_peak_distance)` → `(peak_indices, threshold)`
- `select_keyframes(P, P_smooth, obj_labels, frame_indices, timestamps, peak_indices_per_obj)` → `List[Keyframe]`
- `compute_interaction_probability(cost_path, output_path)` → `(P_smooth, List[Keyframe])`
- `get_keyframe_indices(keyframes)` → `List[int]`

### `keyframes.py`

- `parse_keyframe_table(path)` → `List[Tuple[frame_idx, timestamp_ms, dominant_object, probability]]`
- `extract_keyframe_images(video_path, keyframes, output_dir, jpeg_quality)` → `List[Path]`
- `extract_keyframe_video(video_path, keyframes, output_path, fps)` → `Path`

---

## Mathematical Background

All equations reference the paper included in the repository as `VLM_keyframe_cost_function_PREPRINT.pdf`.

**Hand position** `p_h(t) ∈ R²` is the bounding-box centroid of the detected hand, fused with MediaPipe keypoints for occlusion robustness. Missing detections are forward/backward filled.

**Hand-object distance** (Eq. 1):
```
d_i(t) = ‖p_h(t) − p_oi(t)‖
```

**Cost terms** (Eqs. 2–11) are all normalized to [0, 1] via P90/P10 statistics:

- `phi_d_i(t) = d_i(t) / d_max` — distance cost
- `phi_v(t) = ‖v_h(t)‖ / vh_max` — hand speed cost
- `phi_dir_i(t) = 1 − max(0, cos θ_i(t))` — directional alignment cost
- `phi_obj_i(t)` — piecewise relative velocity cost with dead-zone `v_th`
- `phi_comp_i(t) = 1 / (1 + w_d(t) · α(t))` — finger closure × proximity
- `phi_enc_i(t) = exp(−(ρ_i(t) − 1)) − 1` — convex hull enclosure fraction
- `phi_couple_i(t) = max(0, exp(phi_d · phi_v) − 1)` — nonlinear coupling

**Total cost** (Eq. 12):
```
J_i(t) = phi_d + phi_v + phi_dir + phi_obj + phi_comp + phi_enc + phi_couple
```

**Interaction probability** (Eqs. 13–14): Temperature-scaled softmax over all N objects:
```
P_i(t) = exp(−J_i(t) / τ_i) / Σ_k exp(−J_k(t) / τ_k)

τ_i = σ_{J_i} / (√N / 2)
```
where `σ_{J_i}` is the standard deviation of `J_i` over the full video.

---

## Results and Output Files

After running the full pipeline, the `results/` directory will contain:

```
results/
├── tracking_results.txt          # Raw hand + object positions, frame by frame
├── cost_function.txt             # Seven cost terms + J for each object, each frame
├── interaction_probability.txt   # Smoothed P_i(t), peaks, and keyframe table
└── keyframes/
    ├── kf_001_frame0042_red_block.jpg
    ├── kf_002_frame0107_blue_cup.jpg
    └── ...
```

All `.txt` files use fixed-width columns and a `#`-prefixed comment syntax, making them directly loadable with `numpy.loadtxt()` or `pandas.read_csv(sep=r'\s+', comment='#')` for downstream analysis or plotting.
