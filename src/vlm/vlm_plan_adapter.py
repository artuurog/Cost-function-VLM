"""
VLM-based online plan adapter
===================
"""

import base64
import io
import re
from pathlib import Path

from PIL import Image


# ===========================================================================
# USER SETTINGS  — edit these before running
# ===========================================================================

# --- Backend selection -----------------------------------------------------
# True  -> run the VLM locally with transformers
# False -> call the VLM through the HuggingFace Inference router (API)
USE_LOCAL_INFERENCE = False

# Name of the VLM to use (HuggingFace repository id).
# Must be the same id whether running locally or via the API router.
VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# HuggingFace access token (required for the API backend; also used for gated
# models when running locally). Leave as "" if not needed for a local model.
HF_TOKEN = ""

# --- Input / output paths --------------------------------------------------
WORKSPACE_IMAGE = "workspace.jpg"   # current scene image
DOMAIN_FILE     = "domain.pddl"     # offline task domain
PROBLEM_FILE    = "problem.pddl"    # offline task problem
OUTPUT_PLAN     = "plan.pddl"       # adapted grounded plan (output)

# Object positions in PIXEL coordinates, as produced by the perception module
# (see track_objects.py). Format: {label: (u, v)}.
OBJECT_POSITIONS = {
    # "red_block":   (320, 210),
    # "blue_block":  (455, 198),
    # "green_block": (390, 305),
}

# --- Inference parameters --------------------------------------------------
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
MAX_NEW_TOKENS     = 512
TEMPERATURE        = 0.0            # deterministic output for reproducibility


# ===========================================================================
# Grounded-action grammar
# ---------------------------------------------------------------------------
# This regex is intentionally identical to the one used by the downstream
# bridge (pddl2rapid.py -> PDDLPlanParser._ACTION_RE). Validating the VLM
# output against the very same pattern guarantees that every line we write is
# parseable by the robot bridge.
# ===========================================================================
_ACTION_RE = re.compile(
    r"^\(\s*([a-zA-Z_][a-zA-Z0-9_\-]*)"        # action name
    r"((?:\s+[a-zA-Z_][a-zA-Z0-9_\-]*)*)"      # zero or more identifier params
    r"\s*\)$"
)


# ===========================================================================
# Small helpers
# ===========================================================================

def read_text(path: str) -> str:
    """Read a UTF-8 text file (e.g. a PDDL file)."""
    return Path(path).read_text(encoding="utf-8")


def load_pil_image(path: str) -> Image.Image:
    """Load an image file as an RGB PIL image (used by the local backend)."""
    return Image.open(path).convert("RGB")


def encode_image_base64(path: str) -> str:
    """
    Load an image file and return it as a base64-encoded JPEG string.
    Re-encoding through PIL also normalises PNG/other inputs to JPEG, matching
    the data-URL format expected by the API backend (cf. track_objects.py).
    """
    image = load_pil_image(path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def format_object_positions(object_positions: dict) -> str:
    """Render the detected object positions as 'label: (u, v)' lines."""
    if not object_positions:
        return "(no object positions provided)"
    return "\n".join(f"  {label}: ({u}, {v})"
                     for label, (u, v) in object_positions.items())


# ===========================================================================
# Prompt construction
# ===========================================================================

# Role / behaviour instructions for the VLM. Kept strict so that the raw
# response is directly parseable into a grounded plan.
SYSTEM_INSTRUCTION = (
    "You are a task-plan adaptation module for a robotic manipulator. "
    "You receive a PDDL domain, a PDDL problem, an image of the current "
    "workspace, and the pixel positions of the objects detected in the scene. "
    "Your job is to produce the final grounded plan that achieves the problem "
    "goal in the CURRENT scene. If an object is in a different position than "
    "expected, or if extra objects appear, revise the action sequence "
    "accordingly while respecting the domain actions and the problem goal."
)


def build_adaptation_prompt(domain_text: str,
                            problem_text: str,
                            object_positions: dict) -> str:
    """
    Build the textual prompt for the VLM. The image is attached separately by
    the backend; this function only assembles the symbolic context and the
    strict output-format requirements.
    """
    positions_block = format_object_positions(object_positions)
    return (
        "Adapt the offline task plan to the current workspace shown in the image.\n\n"
        "=== PDDL DOMAIN ===\n"
        f"{domain_text.strip()}\n\n"
        "=== PDDL PROBLEM ===\n"
        f"{problem_text.strip()}\n\n"
        "=== DETECTED OBJECTS (pixel coordinates u, v) ===\n"
        f"{positions_block}\n\n"
        "=== OUTPUT REQUIREMENTS ===\n"
        "- Output ONLY the grounded plan: one action per line.\n"
        "- Each line must have the form (action_name arg1 arg2 ...).\n"
        "- Use ONLY action names defined in the domain.\n"
        "- Use ONLY object and location names from the problem and the detected objects.\n"
        "- Use lowercase names. Do not invent new predicates or arguments.\n"
        "- Do NOT add comments, explanations, reasoning, numbering, or code fences.\n"
        "- The action sequence must be ordered so that it achieves the goal in the current scene.\n"
    )


# ===========================================================================
# VLM backends
# ===========================================================================

def call_vlm_api(image_path: str, prompt: str,
                 model: str, hf_token: str) -> str:
    """
    Query the VLM through the HuggingFace Inference router using the
    OpenAI-compatible client (same pattern as track_objects.py).
    """
    from openai import OpenAI  # imported lazily so local runs don't need it

    if not hf_token:
        raise ValueError("HF_TOKEN is empty but the API backend is selected.")

    client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=hf_token)
    b64_image = encode_image_base64(image_path)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ],
        },
    ]

    print(f"  Calling VLM via HuggingFace router: {model}")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return completion.choices[0].message.content


def call_vlm_local(image_path: str, prompt: str, model: str) -> str:
    """
    Run the VLM locally with HuggingFace transformers.

    Uses the generic image-text-to-text interface, which works for common
    chat VLMs (Qwen-VL, LLaVA-NeXT, SmolVLM, ...). Some models may require a
    specific model/processor class; adjust the imports below if needed.
    """
    import torch  # imported lazily so API runs don't need torch installed
    from transformers import AutoProcessor, AutoModelForImageTextToText

    token = HF_TOKEN or None
    print(f"  Loading local VLM: {model}")
    processor = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, token=token
    )
    vlm = AutoModelForImageTextToText.from_pretrained(
        model, torch_dtype="auto", device_map="auto",
        trust_remote_code=True, token=token,
    )

    image = load_pil_image(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Build the prompt string with the model's chat template, then attach the image.
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(vlm.device)

    print("  Generating adapted plan...")
    with torch.no_grad():
        generated = vlm.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,            # deterministic, mirrors temperature=0
        )

    # Keep only the newly generated tokens (strip the prompt portion).
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def generate_plan_text(image_path: str, prompt: str) -> str:
    """Dispatch to the selected backend and return the raw VLM response."""
    if USE_LOCAL_INFERENCE:
        return call_vlm_local(image_path, prompt, VLM_MODEL)
    return call_vlm_api(image_path, prompt, VLM_MODEL, HF_TOKEN)


# ===========================================================================
# Domain parsing + plan validation
# ===========================================================================

def parse_domain_action_arity(domain_text: str) -> dict:
    """
    Lightweight PDDL domain parser: return {action_name: parameter_count}.

    For each (:action NAME ... :parameters ( ... )) block, the number of
    parameters is the number of '?vars' in the parameters list. This is used
    only to validate the VLM output, so a full PDDL parser is not required.
    """
    arity = {}
    pattern = re.compile(
        r"\(\s*:action\s+([a-zA-Z0-9_\-]+).*?:parameters\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(domain_text):
        name = match.group(1).lower()
        params_blob = match.group(2)
        n_params = len(re.findall(r"\?[a-zA-Z0-9_\-]+", params_blob))
        arity[name] = n_params
    return arity


def extract_plan_actions(raw_text: str) -> list:
    """
    Extract grounded action lines from the raw VLM response.

    Tolerates surrounding prose, code fences, bullets and numbering: only lines
    that match the grounded-action grammar are kept.
    """
    actions = []
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        # Strip common list prefixes (bullets / numbering); the leading '(' of
        # an action is never stripped because it is not in the strip set.
        line = line.lstrip("-*0123456789. ").strip()
        if _ACTION_RE.match(line):
            actions.append(line)
    return actions


def validate_plan(actions: list, action_arity: dict) -> list:
    """
    Check every grounded action against the domain action signatures.
    Returns a list of human-readable warnings (empty if the plan is clean).
    """
    warnings = []
    for action in actions:
        match = _ACTION_RE.match(action)
        name = match.group(1).lower()
        params = match.group(2).split()
        if name not in action_arity:
            warnings.append(f"unknown action '{name}' (not defined in domain): {action}")
        elif action_arity[name] != len(params):
            warnings.append(
                f"action '{name}' arity mismatch: domain expects "
                f"{action_arity[name]} params, got {len(params)}: {action}"
            )
    return warnings


# ===========================================================================
# Output
# ===========================================================================

def write_plan(actions: list, output_path: str, model_name: str) -> None:
    """Write the grounded plan to *output_path* (parseable by pddl2rapid.py)."""
    header = [
        "; plan.pddl - adapted grounded plan",
        f"; generated by the VLM online plan adapter (model: {model_name})",
        "; one grounded action per line; consumed by pddl_to_rapid_bridge.py",
        "",
    ]
    Path(output_path).write_text(
        "\n".join(header + actions) + "\n", encoding="utf-8"
    )
    print(f"  Plan written to: {output_path}  ({len(actions)} actions)")


# ===========================================================================
# Top-level adapter
# ===========================================================================

def adapt_plan(image_path: str,
               object_positions: dict,
               domain_file: str,
               problem_file: str,
               output_plan: str) -> list:
    """
    Run the full online plan adaptation and write the adapted plan.pddl.
    Returns the list of grounded action strings.
    """
    print("=== VLM Online Plan Adapter ===")
    print(f"  Backend : {'local (transformers)' if USE_LOCAL_INFERENCE else 'HuggingFace API'}")
    print(f"  Model   : {VLM_MODEL}")
    print(f"  Image   : {image_path}")
    print(f"  Domain  : {domain_file}")
    print(f"  Problem : {problem_file}")

    domain_text  = read_text(domain_file)
    problem_text = read_text(problem_file)

    # 1) Build the prompt and query the VLM.
    prompt   = build_adaptation_prompt(domain_text, problem_text, object_positions)
    raw_plan = generate_plan_text(image_path, prompt)

    # 2) Extract grounded actions from the raw response.
    actions = extract_plan_actions(raw_plan)
    if not actions:
        print("  [WARNING] No grounded actions found in the VLM response.")
        print("  --- raw VLM output ---")
        print(raw_plan)
        print("  ----------------------")

    # 3) Validate the actions against the domain signatures.
    arity    = parse_domain_action_arity(domain_text)
    warnings = validate_plan(actions, arity)
    for w in warnings:
        print(f"  [WARNING] {w}")

    # 4) Write the adapted plan.
    write_plan(actions, output_plan, VLM_MODEL)
    print("=== Done ===")
    return actions


def main() -> None:
    adapt_plan(
        image_path=WORKSPACE_IMAGE,
        object_positions=OBJECT_POSITIONS,
        domain_file=DOMAIN_FILE,
        problem_file=PROBLEM_FILE,
        output_plan=OUTPUT_PLAN,
    )


if __name__ == "__main__":
    main()