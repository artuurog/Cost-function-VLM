# Learning Long-Horizon Robotic Manipulation from Human Videos via Interaction-Aware Keyframe Selection and Vision-Language Planning

A model-based keyframe extraction pipeline for robotic task learning from human video demonstrations. Given an RGB video of a human manipulation task, this codebase identifies the most informative frames — those corresponding to meaningful hand-object interactions — and delivers them to a Vision-Language Model (VLM) for task plan generation.

This repository implements the methodology described in the paper "Learning Long-Horizon Robotic Manipulation from Human Videos via Interaction-Aware Keyframe Selection and Vision-Language Planning"

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
  - [`src/tracking` — hand and object tracking](#srctracking--hand-and-object-tracking)
  - [`src/keyframes` — cost function and keyframe selection](#srckeyframes--cost-function-and-keyframe-selection)
  - [`src/vlm` — vision-language planning](#srcvlm--vision-language-planning)
  - [`src/bridge` — PDDL to robot execution](#srcbridge--pddl-to-robot-execution)
  - [`src/occlusion` — robustness evaluation](#srcocclusion--robustness-evaluation)
  - [`src/utils` — shared library](#srcutils--shared-library)
  - [`src/robot_code` — ABB RAPID task modules](#srcrobot_code--abb-rapid-task-modules)
  - [`pddl` — reference task definitions](#pddl--reference-task-definitions)
- [Dataset](#dataset)
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
7. The generated PDDL triple (domain, problem, plan) is validated and repaired, again with the VLM as the reasoning core.
8. The validated plan is translated into ABB RAPID instructions and streamed to the robot controller.

<p align="center">
  <img src="media/Graphical abstract.png" alt="Method overview" width="750"/>
</p>


## Repository Structure

```
.
├── src/
│   ├── tracking/                  # Stage 1-2: detect and track hands + objects
│   │   ├── track_hands.py         #   MediaPipe hand tracking (standalone CLI)
│   │   ├── track_objects.py       #   VLM object listing + GroundingDINO detection
│   │   └── track_combined.py      #   Full hand + object tracking pipeline
│   │
│   ├── keyframes/                 # Stage 3-5: cost, probability, keyframe extraction
│   │   ├── cost_function.py       #   Seven-term cost function J_i(t)
│   │   ├── interaction_prob.py    #   Softmax, smoothing, peak picking -> P_i(t)
│   │   └── keyframes.py           #   Extract keyframe images / summary video
│   │
│   ├── vlm/                       # Stage 6-7: vision-language planning
│   │   ├── vlm_learning.py        #   Keyframes -> PDDL domain / problem / plan
│   │   ├── pddl_validator.py      #   Validate + repair a PDDL triple
│   │   └── vlm_plan_adapter.py    #   Online re-grounding of a plan to a new scene
│   │
│   ├── bridge/                    # Stage 8: symbolic plan -> robot motion
│   │   ├── pddl2rapid.py          #   PDDL plan -> ABB RAPID over TCP/IP
│   │   └── pose_config.yaml       #   Symbolic location -> metric pose mapping
│   │
│   ├── occlusion/                 # Robustness evaluation
│   │   └── occlusion.py           #   Synthetic square-patch occlusion of an image
│   │
│   ├── utils/                     # Shared library used by the tracking pipeline
│   │   └── utils.py               #   Data structures, detection, tracking, I/O
│   │
│   └── robot_code/                # ABB RAPID modules, one per demonstrated task
│       ├── BlockStacking.mod
│       ├── BowlStacking.mod
│       ├── Insertion.mod
│       ├── Sorting.mod
│       └── ToolUsage.mod
│
├── pddl/                          # Reference PDDL task definitions
│   ├── bowl/                      #   bowl stacking      (domain + problem)
│   ├── insertion/                 #   bead-on-stick      (domain + problem + plan)
│   ├── sorting/                   #   object sorting     (domain + problem)
│   ├── stacking/                  #   6-block pyramid    (domain + problem + plan)
│   └── tool/                      #   tool usage         (domain + problem + plan)
│
├── media/                         # Figures used in this README and the paper
│   └── Graphical abstract.png
│
└── README.md
```

`keyframes/`, `occlusion/`, `tracking/`, `utils/` and `vlm/` each carry an empty `__init__.py`; `bridge/` does not, and `robot_code/` holds RAPID rather than Python. Scripts are written to be run directly (`python src/<package>/<script>.py`), and each one carries either its own `User settings` block or a command-line interface.

---

## Module Reference

### `src/tracking` — hand and object tracking

| Module | Interface | Purpose |
|---|---|---|
| `track_hands.py` | CLI | Standalone MediaPipe hand tracker. Extracts the 21 hand keypoints, the hand bounding box and the centroid, frame by frame, and optionally writes an annotated video. |
| `track_objects.py` | `User settings` block | Sends the first video frame to the VLM to obtain the list of objects present and their pixel positions, then localizes them with GroundingDINO. |
| `track_combined.py` | `User settings` block | The stage-1/2 entry point. Combines VLM object listing, GroundingDINO detection, multi-object tracking and MediaPipe hand tracking into a single pass, computing relative hand-object distances and velocities. Writes `results/tracking_results.txt`. |

`track_hands.py` is invoked as:

```bash
python src/tracking/track_hands.py --input video.mp4 [--output annotated.mp4] \
    [--no-display] [--skip N] [--max-hands N] \
    [--mp-detect-conf F] [--mp-track-conf F] [--verbose]
```

### `src/keyframes` — cost function and keyframe selection

| Module | Purpose |
|---|---|
| `cost_function.py` | Evaluates the seven cost terms (`phi_d`, `phi_v`, `phi_dir`, `phi_obj`, `phi_comp`, `phi_enc`, `phi_couple`) and their weighted sum `J` for every hand-object pair at every frame. Reads `tracking_results.txt`, writes `cost_function.txt`. |
| `interaction_prob.py` | Converts costs into interaction probabilities with a temperature-scaled softmax, smooths them with a Savitzky-Golay filter, and picks the local maxima above a P90 threshold. Writes `interaction_probability.txt`, including the final keyframe table. |
| `keyframes.py` | Reads the keyframe table and extracts the corresponding video frames, either as one JPEG per keyframe or as a single summary `.mp4` clip. |

These three run in order and each consumes the previous one's output file, so they can be run independently or re-run in isolation after a parameter change.

### `src/vlm` — vision-language planning

All three modules reach the VLM through an OpenAI-compatible endpoint (HuggingFace-style: an API key plus the URL of the model), and all three also support running the model locally through `transformers`.

| Module | Input | Output |
|---|---|---|
| `vlm_learning.py` | The extracted keyframes, in temporal order, each labelled with its dominant object | A PDDL domain, problem and plan describing the demonstrated task |
| `pddl_validator.py` | A PDDL domain, problem and plan | The same three files, revised and corrected, plus a validation report |
| `vlm_plan_adapter.py` | A domain, a problem and an image of the current workspace | A grounded plan re-adapted to the scene as it actually is |

`vlm_learning.py` runs in three stages: per-keyframe grasp/release localisation, plan generation, then a text-only self-critique pass.

`pddl_validator.py` pairs a deterministic symbolic checker with the VLM. The checker parses the three files and gathers evidence — undeclared symbols, arity and type mismatches, unsatisfied preconditions, unreached goals — by simulating the plan forward from the initial state. That evidence is handed to the VLM, which decides what is actually wrong and rewrites the files; the result is re-checked, and the loop repeats until the plan is valid or `MAX_REPAIR_ROUNDS` is exhausted. A revision that is worse than its input is discarded, and the input files are never overwritten. Set `MAX_REPAIR_ROUNDS = 0` to run the static checker alone, with no API calls.

### `src/bridge` — PDDL to robot execution

`pddl2rapid.py` parses a PDDL plan file, maps each grounded action to an ABB RAPID motion instruction, and streams the instructions to the robot controller over a TCP/IP socket:

```bash
python src/bridge/pddl2rapid.py \
    --planner pddl/tool/tool_usage_planner.pddl \
    --config  src/bridge/pose_config.yaml \
    --host    192.168.125.1 \
    --port    5000 \
    [--dry-run] [--delay 0.1] [--verbose]
```

`--dry-run` prints the translated RAPID instructions without opening a connection, which is the way to inspect a plan before it moves a real arm.

`pose_config.yaml` is the symbolic-to-metric bridge: it maps each PDDL location name and orientation descriptor to a robot pose in the base frame — `position` in millimetres, `quaternion` in ABB `[qw, qx, qy, qz]` convention — together with the RAPID `speed_data`, `zone_data` and motion type (`MoveJ` / `MoveL`) to use when reaching it.

### `src/occlusion` — robustness evaluation

`occlusion.py` applies synthetic occlusion to an input image by blacking out `NUM_PATCHES` square patches whose total area covers `OCCLUSION_PERCENTAGE` of the frame. The patch side is computed as `L = sqrt(p·H·W / N)` and positions are sampled uniformly. Used to measure how far keyframe selection and VLM planning degrade as the scene becomes partially hidden.

### `src/utils` — shared library

`utils.py` holds everything the tracking pipeline shares, and is imported by `track_combined.py`:

- **Data structures** — `HandData`, `ObjectData`, `FrameResult`
- **Video** — `extract_first_frame`
- **VLM** — `list_objects_vlm`
- **Detection and tracking** — `detect_objects_dino`, `MultiObjectTracker`, `HandTracker`
- **Kinematics** — `compute_relative_distance`, `compute_relative_velocity`
- **I/O** — `save_results_txt`

### `src/robot_code` — ABB RAPID task modules

One hand-written RAPID module per demonstrated task, loaded onto the ABB controller. They are the execution-side counterpart of the PDDL domains in `pddl/`.

| Module | Task |
|---|---|
| `BlockStacking.mod` | Stack rectangular blocks into a pyramid |
| `BowlStacking.mod` | Nest scattered bowls into one stack, largest at the bottom, without moving the largest |
| `Insertion.mod` | Pick beads from the table and insert each onto its target stick (contact-force step) |
| `Sorting.mod` | Sort objects into their target bins |
| `ToolUsage.mod` | Pick a tool from the rack and apply it to a target |

### `pddl` — reference task definitions

Hand-written reference PDDL for the five tasks, used as ground truth against which the VLM-generated plans are compared, and as input for `pddl_validator.py` and `pddl2rapid.py`.

| Folder | Domain | Problem | Plan |
|---|---|---|---|
| `bowl/` | `bowl_stacking_domain.pddl` | `bowl_stacking_problem.pddl` | — |
| `insertion/` | `insertion_domain.pddl` | `insertion_problem.pddl` | `insertion_plan.pddl` |
| `sorting/` | `sorting_domain.pddl` | `sorting_problem.pddl` | — |
| `stacking/` | `domain.pddl` | `problem.pddl` | `planner.pddl` |
| `tool/` | `tool_usage_domain.pddl` | `tool_usage_problem.pddl` | `tool_usage_planner.pddl` |

The `bowl` and `sorting` folders ship a domain and a problem only; their action sequence is produced by the planner or the VLM.

## Dataset

The dataset containing the human video demonstrations used in this study is available [here](https://huggingface.co/datasets/artuurog/Cost-function-VLM-dataset)

---

## Dependencies

Install all Python dependencies with:

```bash
pip install opencv-python mediapipe openai pillow numpy scipy torch torchvision \
            transformers huggingface_hub pyyaml
```

Roughly, which package serves what:

| Package | Used by |
|---|---|
| `opencv-python` | video I/O and image handling across the pipeline |
| `mediapipe` | hand keypoint tracking |
| `numpy`, `scipy` | cost function, smoothing and peak picking |
| `torch`, `torchvision`, `transformers` | GroundingDINO, and local VLM inference |
| `openai`, `huggingface_hub` | VLM access through the HuggingFace router |
| `pillow` | image encoding for VLM calls |
| `pyyaml` | `pose_config.yaml` in the bridge |

**External model access:**

- A **HuggingFace API token** (`HF_TOKEN` / `API_KEY`) is required for VLM calls. The default model is `allenai/Molmo2-8B`, accessed through `https://router.huggingface.co/v1`.

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
API_KEY = "hf-..."
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
SG_WINDOW_LENGTH  = 11      # Savitzky-Golay window length
SG_POLYORDER      = 3       # Savitzky-Golay polynomial order
MIN_PEAK_DISTANCE = 5       # Minimum separation between probability peaks
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

**`vlm_learning.py`**
```python
INFERENCE_MODE   = "api"    # "api" | "local"
HF_API_KEY       = "hf_..."
MODEL_NAME       = "allenai/Molmo2-8B"
KEYFRAMES_DIR    = "results/keyframes"
OUTPUT_DIR       = "results/pddl"          # where the 3 PDDL files are saved
TASK_NAME        = "sorting_task"
TASK_HINT        = ""                       # optional free-text hint
OBJECT_NAMES     = []                       # override the parsed keyframe labels
ROBOT_SKILLSET   = ["grasp", "move", "drop", "orient"]
LOCALIZE_POINTS  = True     # Stage 1: grasp / release localisation
REFINE_PLAN      = True     # Stage 3: self-critique pass
```

**`pddl_validator.py`**
```python
DOMAIN_FILE  = "pddl/stacking/domain.pddl"     # the 3 inputs
PROBLEM_FILE = "pddl/stacking/problem.pddl"
PLAN_FILE    = "pddl/stacking/planner.pddl"
OUTPUT_DIR   = "results/pddl_revised"          # inputs are never overwritten

API_KEY    = "hf_..."
BASE_URL   = "https://router.huggingface.co/v1"
MODEL_NAME = "allenai/Molmo2-8B"

SCENE_IMAGE       = ""      # optional workspace image; "" = text-only review
MAX_REPAIR_ROUNDS = 2       # 0 = static check only, no API calls
ALWAYS_REVIEW     = True    # call the VLM even when the static check is clean
```

**`vlm_plan_adapter.py`**
```python
USE_LOCAL_INFERENCE = False
VLM_MODEL       = "allenai/Molmo2-8B"
HF_TOKEN        = "hf_..."
WORKSPACE_IMAGE = "workspace.jpg"   # current scene image
DOMAIN_FILE     = "domain.pddl"
PROBLEM_FILE    = "problem.pddl"
OUTPUT_PLAN     = "plan.pddl"
OBJECT_POSITIONS = {}               # {label: (u, v)} detected in the scene
```

**`occlusion.py`**
```python
IMAGE_PATH           = "path/to/frame.jpg"
SAVE_PATH            = "path/to/occluded.jpg"
OCCLUSION_PERCENTAGE = 0.35   # fraction of the image area to cover (0-1)
NUM_PATCHES          = 3      # number of square patches
```

`track_hands.py` and `pddl2rapid.py` take command-line arguments instead; see [Module Reference](#module-reference).

---

## Data Formats

### `results/tracking_results.txt`

One block per frame. Each block starts with a `FRAME` line followed by optional `HAND` and `OBJECT` lines:

```
# FPS: 30.00
FRAME 0 
HAND Right  <cx> <cy>  <kp0x> <kp0y> ... <kp20x> <kp20y>  <bx1> <by1> <bx2> <by2>
OBJECT red_block  <cx> <cy>  <bx1> <by1> <bx2> <by2>
OBJECT blue_cup   <cx> <cy>  <bx1> <by1> <bx2> <by2>
FRAME 1
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

### Keyframe images

`keyframes.py` names each extracted frame using the convention read back by `vlm_learning.py`:

```
keyframe_<rank>_frame<frame_idx>_<object_label>.jpg
```

### PDDL triple

The three PDDL files are plain PDDL, readable by any standard planner. `vlm_learning.py` writes them as `<TASK_NAME>_domain.pddl`, `<TASK_NAME>_problem.pddl` and `<TASK_NAME>_planner.pddl`; `pddl_validator.py` keeps whatever names its inputs had. A plan file is one grounded action per line, with `;` comments allowed:

```
(grasp   gofa_robot block_1 table_slot_1)
(move    gofa_robot block_1 loc_L1_P1)
(release gofa_robot block_1 loc_L1_P1)
```

The timestamped IPC format (`0.001: (grasp gofa_robot block_1 table_slot_1) [1.000]`) is also accepted by the validator.

---

## Results and Output Files

After running the full pipeline, the `results/` directory will contain:

```
results/
├── tracking_results.txt          # Raw hand + object positions, frame by frame
├── cost_function.txt             # Seven cost terms + J for each object, each frame
├── interaction_probability.txt   # Smoothed P_i(t), peaks, and keyframe table
├── keyframes/
│   ├── kf_001_frame0042_red_block.jpg
│   ├── kf_002_frame0107_blue_cup.jpg
│   └── ...
├── pddl/                         # Generated by vlm_learning.py
│   ├── <task>_domain.pddl
│   ├── <task>_problem.pddl
│   └── <task>_planner.pddl
└── pddl_revised/                 # Generated by pddl_validator.py
    ├── <domain>.pddl             #   the three files, revised and corrected
    ├── <problem>.pddl
    ├── <plan>.pddl
    └── validation_report.txt     #   findings, and whether the plan is valid
```

All `.txt` files use fixed-width columns and a `#`-prefixed comment syntax, making them directly loadable with `numpy.loadtxt()` or `pandas.read_csv(sep=r'\s+', comment='#')` for downstream analysis or plotting.
