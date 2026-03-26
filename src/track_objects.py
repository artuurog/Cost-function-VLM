"""

"""

import base64
import io
import sys
from pathlib import Path

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
    "For each object say its name."
    "Select a one pixel for each object on the table. The object pixels must be written next to the name of the object."
    "\nExample output:"
    "\ngreen cup: (100, 200)"
    "\nwhite cup: (200, 300)"
    "\nred cup: (300, 400)"
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
# STEP 2 — Send the frame to Molmo-2-8B via the HuggingFace API
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


def call_molmo(frame, prompt: str, hf_token: str) -> str:
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
        model="allenai/Molmo2-8B",
        # model="Qwen/Qwen3-VL-8B-Instruct",
        messages=messages,
        max_tokens=512,
    )

    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

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

    response = call_molmo(frame, PROMPT, HF_TOKEN)

    print("\n" + "-" * 55)
    print("  MODEL RESPONSE:")
    print("-" * 55)
    print(response)
    print("-" * 55)

    print("\n VLM object localization complete.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()