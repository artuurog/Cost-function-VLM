"""
VLM for task learning
==============
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# User settings — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

INFERENCE_MODE: str = "api"

# --- HuggingFace API settings (used only when INFERENCE_MODE == "api") ---------
HF_API_KEY: str = ""              # your HuggingFace access token ("hf_...")
HF_PROVIDER: Optional[str] = None  # e.g. "hf-inference", "together"; None = auto

MODEL_NAME: str = "allenai/Molmo2-8B"

# --- I/O ----------------------------------------------------------------------
KEYFRAMES_DIR: str = "results/keyframes"          # directory with keyframe
PROBABILITY_FILE: str = "results/interaction_probability.txt"  # optional fallback
OUTPUT_DIR: str = "results/pddl"                  # where the 3 PDDL files are saved

# Short identifier used to name the output files and the PDDL (domain problem).
TASK_NAME: str = "sorting_task"

TASK_HINT: str = ""

OBJECT_NAMES: List[str] = []

# --- Planning behaviour -------------------------------------------------------
ROBOT_SKILLSET: List[str] = ["grasp", "move", "drop", "orient"]

LOCALIZE_POINTS: bool = True   
REFINE_PLAN: bool = True     

# --- Generation settings ------------------------------------------------------
MAX_NEW_TOKENS: int = 2048
TEMPERATURE: float = 0.2

# Longest image side (px) sent to the VLM; keeps token / bandwidth cost bounded.
MAX_IMAGE_SIDE: int = 768


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    """A single keyframe and the object the hand interacts with in it."""
    frame_idx: int
    object_label: str
    image_path: Path
    notes: str = ""             # grasp / release localisation, filled in stage 1
    _image: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def object_text(self) -> str:
        """Human-readable object name (underscores → spaces) for prompting."""
        return self.object_label.replace("_", " ")

    def image(self) -> Image.Image:
        if self._image is None:
            img = Image.open(self.image_path).convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            self._image = img
        return self._image


# A unified message format shared by both backends:
#   {"role": str, "content": [ {"type": "text", "text": str}
#                            | {"type": "image", "image": PIL.Image} ]}
Message = Dict[str, object]


# ─────────────────────────────────────────────────────────────────────────────
# Keyframe loading
# ─────────────────────────────────────────────────────────────────────────────

# Filename convention written by keyframes.py:
#   keyframe_<rank>_frame<frame_idx>_<safe_label>.jpg
_KEYFRAME_RE = re.compile(
    r"keyframe_\d+_frame(?P<idx>\d+)_(?P<label>.+)\.jpe?g$",
    re.IGNORECASE,
)


def load_keyframes(
    keyframes_dir: str,
    object_names: Optional[List[str]] = None,
) -> List[Keyframe]:
    """
    Load keyframes from *keyframes_dir* in temporal order (ascending frame index).

    The object label of each keyframe is taken from *object_names* when supplied,
    otherwise parsed from the filename.
    """
    directory = Path(keyframes_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Keyframes directory not found: '{directory}'")

    parsed: List[Keyframe] = []
    for path in directory.iterdir():
        m = _KEYFRAME_RE.match(path.name)
        if m is None:
            continue
        parsed.append(
            Keyframe(
                frame_idx=int(m.group("idx")),
                object_label=m.group("label"),
                image_path=path,
            )
        )

    if not parsed:
        raise FileNotFoundError(
            f"No keyframe images matching 'keyframe_*_frame*_*.jpg' in '{directory}'."
        )

    # Temporal order is essential for correct action sequencing.
    parsed.sort(key=lambda kf: kf.frame_idx)

    if object_names:
        if len(object_names) != len(parsed):
            raise ValueError(
                f"OBJECT_NAMES has {len(object_names)} entries but "
                f"{len(parsed)} keyframes were found."
            )
        for kf, name in zip(parsed, object_names):
            kf.object_label = name.strip().replace(" ", "_")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# VLM backends
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG data URL (for the API backend)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class VLMBackend:
    """Abstract VLM backend: chat(messages) -> assistant text."""

    def chat(self, messages: List[Message]) -> str:
        raise NotImplementedError


class APIBackend(VLMBackend):
    """HuggingFace Inference API backend (OpenAI-style chat completions)."""

    def __init__(self) -> None:
        from huggingface_hub import InferenceClient

        if not HF_API_KEY:
            raise ValueError("HF_API_KEY is empty; set it to your HuggingFace token.")
        self._client = InferenceClient(api_key=HF_API_KEY, provider=HF_PROVIDER)

    def chat(self, messages: List[Message]) -> str:
        oai_messages = [self._to_openai(m) for m in messages]
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=oai_messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _to_openai(message: Message) -> Dict[str, object]:
        """Convert the unified message format to OpenAI multimodal content."""
        content = []
        for part in message["content"]:  # type: ignore[index]
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:  # image
                url = _image_to_data_url(part["image"])
                content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": message["role"], "content": content}


class LocalBackend(VLMBackend):
    """
    Local transformers backend.

    """

    def __init__(self) -> None:
        import torch  # noqa: F401  (imported lazily; used by generate)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

    def chat(self, messages: List[Message]) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            )

        # Drop the prompt tokens, keep only the newly generated answer.
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()


def build_backend() -> VLMBackend:
    """Instantiate the backend selected by INFERENCE_MODE."""
    if INFERENCE_MODE == "api":
        return APIBackend()
    if INFERENCE_MODE == "local":
        return LocalBackend()
    raise ValueError(f"Unknown INFERENCE_MODE '{INFERENCE_MODE}'. Use 'api' or 'local'.")


# ─────────────────────────────────────────────────────────────────────────────
# Message helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text(role: str, text: str) -> Message:
    return {"role": role, "content": [{"type": "text", "text": text}]}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def _system_prompt() -> str:
    skills = ", ".join(ROBOT_SKILLSET)
    return (
        "You are a task-planning expert for a robotic manipulator. You are given "
        "a series of keyframes taken from a single human manipulation demonstration, in temporal "
        "order, together with the label of the object the hand interacts with in "
        "each keyframe. Your job is to infer the demonstrated task and express it "
        "as a PDDL task plan that the robot can execute.\n\n"
        f"The robot can only perform these primitive actions: {skills}. Build the "
        "domain actions out of this skillset.\n\n"
        "Produce three PDDL components and wrap each one in the exact delimiters "
        "shown below, with valid PDDL inside and nothing else between the tags:\n"
        "<DOMAIN>\n(define (domain ...) ...)\n</DOMAIN>\n"
        "<PROBLEM>\n(define (problem ...) (:domain ...) ...)\n</PROBLEM>\n"
        "<PLANNER>\n(one grounded action per line, in execution order)\n</PLANNER>\n\n"
        "Rules:\n"
        "- DOMAIN: declare object types, predicates (robot state, object state, "
        "spatial layout) and the parametric actions with parameters, preconditions "
        "and effects.\n"
        "- PROBLEM: list the concrete objects and target locations, the initial "
        "state, and the goal configuration that reproduces the demonstration.\n"
        "- PLANNER: the ordered sequence of grounded actions that reaches the goal, "
        "one action per line, e.g. (grasp robot block_red).\n"
        "- Keep symbols consistent across the three components and grounded in the "
        "provided object labels. Use symbolic location names (not metric coordinates); "
        "metric poses are resolved later by the robot controller."
    )


def _localization_prompt(keyframe: Keyframe) -> List[Message]:
    """Stage 1 prompt: localize grasp and release points in one keyframe."""
    instruction = (
        f"This keyframe shows the hand interacting with the '{keyframe.object_text}'. "
        "Point to (a) the grasp point — the pixel where the fingertips contact the "
        "object — and (b) the release / target point — where the object is being "
        "placed. Answer concisely as: grasp=(x,y); release=(x,y); plus a short note "
        "on what action is happening."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": keyframe.image()},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def _planning_message(keyframes: List[Keyframe]) -> Message:
    """Stage 2 user message: interleave each keyframe image with its labelled text."""
    content: List[Dict[str, object]] = []

    header = "Here is the demonstration, keyframe by keyframe in temporal order."
    if TASK_HINT:
        header += f" Task hint: {TASK_HINT}"
    content.append({"type": "text", "text": header})

    for step, kf in enumerate(keyframes, start=1):
        content.append({"type": "image", "image": kf.image()})
        line = (
            f"Keyframe {step} (frame {kf.frame_idx}): the hand interacts with "
            f"'{kf.object_text}'."
        )
        if kf.notes:
            line += f" Localisation: {kf.notes}"
        content.append({"type": "text", "text": line})

    content.append(
        {
            "type": "text",
            "text": (
                "Now infer the task and output the DOMAIN, PROBLEM and PLANNER "
                "components in the required delimited format."
            ),
        }
    )
    return {"role": "user", "content": content}


def _refine_prompt(draft: str) -> str:
    """Stage 3 prompt: text-only self-critique and refinement."""
    return (
        "Critically review the PDDL task plan you produced (below). Check for "
        "logical inconsistencies and missing preconditions between consecutive "
        "actions, unreachable goals, and symbol mismatches across the three "
        "components. Do NOT re-examine any images; reason only on the text. Then "
        "output the corrected DOMAIN, PROBLEM and PLANNER using the same delimited "
        "format. If the plan is already correct, return it unchanged.\n\n"
        f"{draft}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def localize_contact_points(backend: VLMBackend, keyframes: List[Keyframe]) -> None:
    """Stage 1: fill keyframe.notes with grasp / release localisation per frame."""
    for kf in keyframes:
        try:
            kf.notes = backend.chat(_localization_prompt(kf)).strip()
        except Exception as exc:  # localisation is auxiliary; never abort the run
            print(f"  [WARN] localisation failed for frame {kf.frame_idx}: {exc}")
            kf.notes = ""


def generate_task_plan(backend: VLMBackend, keyframes: List[Keyframe]) -> str:
    """Stage 2: produce the initial DOMAIN / PROBLEM / PLANNER draft."""
    messages: List[Message] = [
        _text("system", _system_prompt()),
        _planning_message(keyframes),
    ]
    return backend.chat(messages)


def refine_task_plan(backend: VLMBackend, draft: str) -> str:
    """Stage 3: self-critique and refine the draft (text-only)."""
    messages: List[Message] = [
        _text("system", _system_prompt()),
        _text("user", _refine_prompt(draft)),
    ]
    return backend.chat(messages)


# ─────────────────────────────────────────────────────────────────────────────
# PDDL extraction and output
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_RE = {
    "domain": re.compile(r"<DOMAIN>\s*(.*?)\s*</DOMAIN>", re.DOTALL | re.IGNORECASE),
    "problem": re.compile(r"<PROBLEM>\s*(.*?)\s*</PROBLEM>", re.DOTALL | re.IGNORECASE),
    "planner": re.compile(r"<PLANNER>\s*(.*?)\s*</PLANNER>", re.DOTALL | re.IGNORECASE),
}

# Strip any markdown code fences the model may wrap the PDDL in.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def parse_pddl_sections(text: str) -> Dict[str, str]:
    """Extract the three PDDL components from the delimited VLM output."""
    sections: Dict[str, str] = {}
    for name, pattern in _SECTION_RE.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(
                f"Could not find a <{name.upper()}> ... </{name.upper()}> section "
                f"in the VLM output. Raw output:\n{text}"
            )
        sections[name] = _FENCE_RE.sub("", match.group(1)).strip()
    return sections


def write_pddl_files(sections: Dict[str, str], output_dir: str, task_name: str) -> Dict[str, Path]:
    """Write domain / problem / planner to <output_dir>/<task>_<section>.pddl."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, content in sections.items():
        path = out / f"{task_name}_{name}.pddl"
        path.write_text(content + "\n", encoding="utf-8")
        written[name] = path
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run() -> Dict[str, Path]:

    print(f"[INFO] Loading keyframes from '{KEYFRAMES_DIR}' ...")
    keyframes = load_keyframes(KEYFRAMES_DIR, OBJECT_NAMES or None)
    print(f"[INFO] {len(keyframes)} keyframe(s) loaded (temporal order):")
    for step, kf in enumerate(keyframes, start=1):
        print(f"    {step:2d}. frame {kf.frame_idx:5d}  object='{kf.object_text}'")

    print(f"[INFO] Building VLM backend (mode='{INFERENCE_MODE}', model='{MODEL_NAME}') ...")
    backend = build_backend()

    if LOCALIZE_POINTS:
        print("[INFO] Stage 1 — localizing grasp / release points ...")
        localize_contact_points(backend, keyframes)

    print("[INFO] Stage 2 — generating initial task plan ...")
    draft = generate_task_plan(backend, keyframes)

    final = draft
    if REFINE_PLAN:
        print("[INFO] Stage 3 — self-critique and refinement ...")
        final = refine_task_plan(backend, draft)

    print("[INFO] Parsing PDDL sections ...")
    sections = parse_pddl_sections(final)

    written = write_pddl_files(sections, OUTPUT_DIR, TASK_NAME)
    print("[DONE] PDDL files written:")
    for name, path in written.items():
        print(f"    {name:8s} → {path}")
    return written


if __name__ == "__main__":
    run()