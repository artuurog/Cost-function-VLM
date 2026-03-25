"""
step1_extract_frame.py
======================
Usage
-----
  python track_objects.py --video path/to/video.mp4 --hf-token hf_xxxx

"""

import argparse
import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default Molmo-2-8B endpoint on the HuggingFace Inference API.
# We only validate the token here; the actual API call comes in Step 2.
MOLMO_ENDPOINT = (
    "https://api-inference.huggingface.co/models/allenai/Molmo-2-8B-O-0924"
)

HF_TOKEN = "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"
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_first_frame(video_path: str) -> tuple:
    """
    Open the video and read the very first frame.

    Returns
    -------
    frame : numpy ndarray, shape (H, W, 3), dtype uint8, BGR colour order
    fps   : float — frames per second of the source video
    total : int   — total number of frames in the video
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video file: '{video_path}'\n"
            "Make sure the path is correct and the file is a valid .mp4."
        )

    # Read basic video metadata before grabbing the frame
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(
            "The video opened successfully but the first frame could not be read.\n"
            "The file may be corrupted or encoded in an unsupported format."
        )

    print(f"  Resolution : {width} x {height} pixels")
    print(f"  Frame rate : {fps:.2f} fps")
    print(f"  Duration   : {total} frames  (~{total / fps:.1f} seconds)")

    return frame, fps, total


def save_frame(frame, output_path: str) -> None:
    """Save a BGR frame as a JPEG image to disk."""
    success = cv2.imwrite(output_path, frame)
    if not success:
        raise RuntimeError(f"cv2.imwrite failed for path: '{output_path}'")
    print(f"  First frame saved to: {output_path}")


def show_frame(frame, window_title: str = "First Frame — press any key to close") -> None:
    """
    Display the frame in an OpenCV window.
    The window stays open until the user presses any key.
    Works on desktop environments; in headless servers it is skipped gracefully.
    """
    try:
        cv2.imshow(window_title, frame)
        cv2.waitKey(0)          # wait indefinitely for a key press
        cv2.destroyAllWindows()
    except cv2.error:
        # OpenCV raises an error when there is no display available (e.g. SSH)
        print(
            "  [INFO] No display detected — skipping interactive window.\n"
            "  Open the saved JPEG file to view the frame."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 — Extract and display the first frame of an .mp4 video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input .mp4 video file.",
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="Your HuggingFace API token (starts with 'hf_'). "
             "Required now so the full pipeline can reuse this script.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Validate inputs ---------------------------------------------------

    video_path = args.video

    if not Path(video_path).is_file():
        print(f"[ERROR] Video file not found: '{video_path}'")
        sys.exit(1)

    if not HF_TOKEN.startswith("hf_"):
        print("[WARNING] The HuggingFace token usually starts with 'hf_'. "
              "Double-check that it is correct.")

    print("\n" + "=" * 55)
    print("  STEP 1 — Extract first frame")
    print("=" * 55)

    # --- Pipeline configuration summary -----------------------------------
    print(f"\n  Video file    : {video_path}")
    print(f"  Molmo endpoint: {MOLMO_ENDPOINT}")

    # --- Extract the first frame ------------------------------------------
    print("\n  Extracting first frame...")
    frame, fps, total_frames = extract_first_frame(video_path)

    # --- Save the frame to disk -------------------------------------------
    # Place the output JPEG next to the input video, same name + suffix
    stem        = Path(video_path).stem
    output_dir  = Path(video_path).parent
    output_path = str(output_dir / f"{stem}_first_frame.jpg")

    save_frame(frame, output_path)

    # --- Display the frame ------------------------------------------------
    print("\n  Displaying frame (press any key in the window to close)...")
    show_frame(frame)

    # --- Done -------------------------------------------------------------
    print("\n  Step 1 complete.")
    print(f"  Next step: send '{output_path}' to Molmo-2-8B for object detection.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()