import numpy as np
import cv2


# Path to the input RGB image
IMAGE_PATH = "C:/Users/user/Desktop/VLM keyframe/immagini VLM keyframe/annotated_hands/sorting_demo (1).png"

OCCLUSION_PERCENTAGE = 0.15

NUM_PATCHES = 3

# ---------------------------------------------------------------------------


def compute_patch_side(H: int, W: int, occlusion_percentage: float,
                        num_patches: int) -> int:
    
    L = int(np.sqrt(occlusion_percentage * H * W / num_patches))
    return L


def sample_patch_positions(H: int, W: int, L: int,
                            num_patches: int) -> list[tuple[int, int]]:
    positions = []
    for _ in range(num_patches):
        # Ensure the patch fits inside the image
        row = np.random.randint(0, H - L)
        col = np.random.randint(0, W - L)
        positions.append((row, col))
    return positions


def apply_patches(image: np.ndarray, positions: list[tuple[int, int]],
                  L: int) -> np.ndarray:
   
    occluded = image.copy()
    for (row, col) in positions:
        occluded[row:row + L, col:col + L] = 0   # set pixels to black
    return occluded


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")

    # ------------------------------------------------------------------
    # 2. Image dimensions
    # ------------------------------------------------------------------
    H, W = image.shape[:2]
    print(f"Image size: H={H}, W={W}")

    # ------------------------------------------------------------------
    # 3. Read occlusion percentage from user parameter (already a float)
    # ------------------------------------------------------------------
    occlusion_pct = float(OCCLUSION_PERCENTAGE)
    assert 0.0 < occlusion_pct < 1.0, \
        "OCCLUSION_PERCENTAGE must be strictly between 0 and 1."

    # ------------------------------------------------------------------
    # 4. Compute patch side length L
    # ------------------------------------------------------------------
    L = compute_patch_side(H, W, occlusion_pct, NUM_PATCHES)
    print(f"Occlusion percentage : {occlusion_pct * 100:.1f} %")
    print(f"Patch side length    : L = {L} px  ({L}×{L} pixels per patch)")
    print(f"Actual covered area  : "
          f"{NUM_PATCHES * L * L / (H * W) * 100:.2f} % "
          f"(target: {occlusion_pct * 100:.1f} %)")

    # ------------------------------------------------------------------
    # 5. Sample 3 random patch positions
    # ------------------------------------------------------------------
    np.random.seed(None)   # remove or set a fixed seed for reproducibility
    positions = sample_patch_positions(H, W, L, NUM_PATCHES)
    for idx, (r, c) in enumerate(positions):
        print(f"  Patch {idx + 1}: top-left=({r}, {c}), "
              f"bottom-right=({r + L}, {c + L})")

    # ------------------------------------------------------------------
    # 6. Apply patches to a copy of the image
    # ------------------------------------------------------------------
    occluded = apply_patches(image, positions, L)

    # ------------------------------------------------------------------
    # 7. Display original and occluded images side by side
    # ------------------------------------------------------------------
    cv2.imshow("Original image", image)
    cv2.imshow("Occluded image", occluded)
    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()