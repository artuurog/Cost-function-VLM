"""

"""

import base64
import io
import sys
from pathlib import Path
import re

import cv2
from openai import OpenAI
from PIL import Image


# ---------------------------------------------------------------------------
# USER SETTINGS  — edit these before running
# ---------------------------------------------------------------------------

VIDEO_PATH = "C:/Users/user/Desktop/PoliMi/DOTTORATO/hand object interaction/video/my_demos/red_block1.mp4"
HF_TOKEN   = "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"

PROMPT = (
    "You are an expert object localizer. Point to all the objects on the black table. "
    "For each object assign a label. If two objects are visually similar, assign a label that reflects their size (big/small)."
    "Object labels must be unique."
    "Select a one pixel for each object on the table. The position of the object must be written in pixel coordinates next to its label."
    "\n The output must strictly follow the following structure for each identified object:"
    "\n OBJECT_NAME: (pixel_u, pixel_v)"
    # "\n small green cup 1: (100, 200)"
    # "\n small green cup 2: (250, 420)"
    # "\n small white cup: (200, 300)"
    # "\n big white cup: (500, 540)"
    # "\n red cup: (300, 400)"
    "\n Do not show your reasoning. Write only the list of labels and pixels."
)


# ---------------------------------------------------------------------------
# STEP 1 — Extract the first frame from the video
# ---------------------------------------------------------------------------

def extract_first_frame(video_path: str):
    """
    Open the video file and read the very first frame.

    Returns
    -------
    frame  : numpy array (H, W, 3) in BGR colour order
    fps    : frames per second of the video
    total  : total number of frames
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: '{video_path}'")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Could not read the first frame from the video.")

    print(f"  Resolution : {width} x {height} pixels")
    print(f"  Frame rate : {fps:.2f} fps")
    print(f"  Duration   : {total} frames (~{total / fps:.1f} seconds)")

    return frame, fps, total


def save_frame(frame, output_path: str) -> None:
    """Save a BGR frame as a JPEG file."""
    cv2.imwrite(output_path, frame)
    print(f"  Frame saved to: {output_path}")


def show_frame(frame) -> None:
    """Display the frame in a window. Press any key to close."""
    try:
        cv2.imshow("First Frame — press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  [INFO] No display available. Open the saved JPEG to view the frame.")


# ---------------------------------------------------------------------------
# Send the frame to VLM via HuggingFace API
# ---------------------------------------------------------------------------

def encode_frame_as_base64(frame) -> str:
    """
    Convert a BGR OpenCV frame to a base64-encoded JPEG string.
    """
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    b64_string = base64.b64encode(buffer.read()).decode("utf-8")
    return b64_string


def call_vlm(frame, prompt: str, hf_token: str) -> str:
    """
    Send the image + prompt to Molmo-2-8B via the HuggingFace router.
    """

    # --- Build the client -------------------------------------------------
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

    # --- Encode the frame -------------------------------------------------
    print("  Encoding frame as base64 JPEG...")
    b64_image = encode_frame_as_base64(frame)

    # --- Build the message ------------------------------------------------
    # The message has two parts: the text prompt and the image.
    # The image is embedded as a data URI so no external URL is needed.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    },
                },
            ],
        }
    ]

    # --- Call the API -----------------------------------------------------
    print(f"  Prompt  : \"{prompt}\"")
    print("  Calling VLM...")

    completion = client.chat.completions.create(
        # model="allenai/Molmo2-8B",
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=messages,
        max_tokens=300,
        temperature=0,
    )

    return completion.choices[0].message.content


def get_obj_coordinates(response: str) -> tuple[list[tuple[int, int]], list[str]]:
    
    objects     = []
    coordinates = []
 
    # Each line looks like:   <name>: (<u>, <v>)
    # The pattern captures:
    #   group 1 — the object name  (anything before the colon)
    #   group 2 — u coordinate     (integer inside the parentheses)
    #   group 3 — v coordinate     (integer inside the parentheses)
    pattern = re.compile(r"^(.+?):\s*\((\d+),\s*(\d+)\)", re.MULTILINE)
 
    for match in pattern.finditer(response):
        name = match.group(1).strip()
        u    = int(match.group(2))
        v    = int(match.group(3))
 
        objects.append(name)
        coordinates.append((u, v))
 
    return coordinates, objects
 

def annotated_frame(frame, coords:list[tuple[int, int]] , obj_list:list[str] ) -> None:
    annotated = frame.copy()
 
    for (u, v), label in zip(coords, obj_list):
 
        # --- Draw circle at the object location ------------
        cv2.circle(
            annotated,
            center=(u, v),
            radius=6,
            color=(0, 255, 0),   
            thickness=-1
        )
 
        # --- Draw the label text --
        cv2.putText(
            annotated,
            text=label,
            org=(u + 10, v - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
 
    # --- Display the annotated frame --------------------------------------
    try:
        cv2.imshow("Annotated Frame — press any key to close", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  [INFO] No display available. Open the saved JPEG to view the frame.")

    
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Check that the video file exists ---------------------------------
    if not Path(VIDEO_PATH).is_file():
        print(f"[ERROR] Video not found: '{VIDEO_PATH}'")
        sys.exit(1)

    # =========================================================
    # Extract the first frame
    # =========================================================
    print("\n" + "=" * 55)
    print(" Extract first frame")
    print("=" * 55)
    print(f"\n  Video : {VIDEO_PATH}\n")

    frame, fps, total_frames = extract_first_frame(VIDEO_PATH)

    # Save the frame as a JPEG in the same folder as the video
    # stem       = Path(VIDEO_PATH).stem
    # output_dir = Path(VIDEO_PATH).parent
    # frame_path = str(output_dir / f"{stem}_first_frame.jpg")

    # save_frame(frame, frame_path)
    show_frame(frame)

    print("\n  Frame visualized.\n")

    # =========================================================
    # VLM frame analysis
    # =========================================================
    print("=" * 55)
    print("  VLM object localization")
    print("=" * 55 + "\n")

    response = call_vlm(frame, PROMPT, HF_TOKEN)

    print("\n" + "-" * 55)
    print("  MODEL RESPONSE:")
    print("-" * 55)
    print(response)
    print("-" * 55)
    
    coords, obj_list = get_obj_coordinates(response)
    
    print("Recognized objects:")
    print(obj_list)
    print("Pixel positions:")
    print(coords)
    
    # =========================================================
    # Visualize VLM result
    # =========================================================
    
    annotated_frame(frame, coords, obj_list)

    print("\n VLM object localization complete.")
    print("=" * 55 + "\n")