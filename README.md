# Learning Long-Horizon Robotic Manipulation from Human Videos via Interaction-Aware Keyframe Selection and Vision-Language Planning

A model-based keyframe extraction pipeline for robotic task learning from human video demonstrations. Given an RGB video of a human manipulation task, this codebase identifies the most informative frames — those corresponding to meaningful hand-object interactions — and delivers them to a Vision-Language Model (VLM) for task plan generation.

This repository implements the methodology described in the paper "Learning Long-Horizon Robotic Manipulation from Human Videos via Interaction-Aware Keyframe Selection and Vision-Language Planning"

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Data Formats](#data-formats)
- [Results and Output Files](#results-and-output-files)

---

## Overview

The pipeline proceeds as follows:

1. A VLM identifies all objects in the first frame of the video.
2. An open-vocabulary object detector (GroundingDINO) and a hand tracker (MediaPipe) localize all entities frame by frame.
3. A seven-term cost function `J_i(t)` is evaluated for every hand-object pair `(hand, object_i)` at every frame `t`.
4. A temperature-scaled softmax converts the costs into interaction probabilities `P_i(t)`.
5. The frames corresponding to local probability maxima are extracted as keyframes, each annotated with the dominant interacting object.
6. The keyframes are fed sequentially to a VLM for task understanding and PDDL plan generation.

## Repo Structure

```
.           
├── src/
|   ├── bridge
|       ├── pddl2rapid.py
|       └── pose_config.yaml
|   ├── keyframes
|       ├── cost_function.py
|       ├── keyframes.py
|       └── interaction_prob.py
|   ├── tracking
|       ├── track_combined.pyù
|       ├── track_hands.py
|       └── track_objects.py
|   └── utils
|       └── utils.yaml
├── pddl/
|   ├── bowl/
|   ├── insertion/
|   ├── sorting/
|   ├── stacking/
|   └── tool/
└── 
```

---

## Dependencies

Install all Python dependencies with:

```bash
pip install opencv-python mediapipe openai pillow numpy scipy torch torchvision transformers
```

**External model access:**

- A **HuggingFace API token** (`HF_TOKEN`) is required for VLM calls. The default model is `allenai/Molmo2-8B`, accessed through `https://router.huggingface.co/v1`.

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
```

### `results/interaction_probability.txt`

Contains per-object probability time histories followed by the final keyframe table:

```
# OBJECT: red_block
#   P90 threshold used for peak filtering: 0.412300
#    frame_idx  timestamp_ms    P_raw   P_smooth
          ...

# KEYFRAME TABLE
#     rank  frame_idx  timestamp_ms        dominant_object  probability

```

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
