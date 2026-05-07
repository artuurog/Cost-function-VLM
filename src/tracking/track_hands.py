"""
Video Hand-Object Interaction Tracker
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Setup MediaPipe
# ──────────────────────────────────────────────────────────────────────────────

MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles

# Hand keypoint indeces
KEYPOINT_NAMES = [
    "WRIST",                                           # 0
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",  # 1-4
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",  # 5-8
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",  # 9-12
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",   # 13-16
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",  # 17-20
]

# fingertip indeces
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

class HandData:
    """Hand data from a single frame."""

    def __init__(self, handedness: str, keypoints: np.ndarray, bbox: tuple):
        """
        Args:
            handedness: "Left" or "Right"
            keypoints:  Array (21, 2)
            bbox:       (x1, y1, x2, y2)
        """
        self.handedness = handedness          # "Left" / "Right"
        self.keypoints = keypoints            # shape (21, 2), float32
        self.bbox = bbox                      # (x1, y1, x2, y2) in pixel

    @property
    def wrist(self) -> np.ndarray:
        """Wrist (keypoint 0)."""
        return self.keypoints[0]

    @property
    def centroid(self) -> np.ndarray:
        """bbox center"""
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    @property
    def fingertips(self) -> np.ndarray:
        """Fingertip coordinates, shape (5, 2)."""
        return self.keypoints[FINGERTIP_INDICES]

    def hand_spread(self) -> float:
        dists = np.linalg.norm(self.keypoints[1:] - self.wrist, axis=1)
        return float(np.mean(dists))


class FrameResult:
    """Results from the analysis of a single frame."""

    def __init__(self, frame_idx: int, timestamp_ms: float,
                 hands: list[HandData]):
        self.frame_idx = frame_idx
        self.timestamp_ms = timestamp_ms
        self.hands = hands      # Lista di HandData


# ──────────────────────────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────────────────────────

class VideoHOITracker:
    """
    Tracker Hand-Object Interaction for MP4 video.

    """

    def __init__(
        self,
        mp_max_num_hands: int = 2,
        mp_min_detection_confidence: float = 0.5,
        mp_min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            mp_max_num_hands
            mp_min_detection_confidence
            mp_min_tracking_confidence
        """
        # MediaPipe Hands
        self.mp_hands = MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=mp_max_num_hands,
            min_detection_confidence=mp_min_detection_confidence,
            min_tracking_confidence=mp_min_tracking_confidence,
        )

        

    def _detect_hands(self, frame_rgb: np.ndarray, frame_w: int, frame_h: int) -> list[HandData]:

        results = self.mp_hands.process(frame_rgb)
        hands = []

        if not results.multi_hand_landmarks:
            return hands

        for lm_list, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            handedness = handedness_info.classification[0].label  # "Left" / "Right"

            # extract 21 keypoints in pixel values
            keypoints = np.array(
                [[lm.x * frame_w, lm.y * frame_h] for lm in lm_list.landmark],
                dtype=np.float32
            )  # shape: (21, 2)

            # compute bbox of points
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            pad = 15  # pixel  padding
            x1 = max(0, int(np.min(x_coords)) - pad)
            y1 = max(0, int(np.min(y_coords)) - pad)
            x2 = min(frame_w, int(np.max(x_coords)) + pad)
            y2 = min(frame_h, int(np.max(y_coords)) + pad)

            hands.append(HandData(
                handedness=handedness,
                keypoints=keypoints,
                bbox=(x1, y1, x2, y2),
            ))

        return hands

    def process_video(
        self,
        input_path: str,
        output_path: str | None = None,
        display: bool = True,
        skip_frames: int = 0,
    ) -> list[FrameResult]:
        """
        Args:
            input_path
            output_path
            display
            skip_frames
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {input_path}")
        print(f"[INFO] Resolution: {frame_w}x{frame_h} @ {fps:.1f} fps")
        print(f"[INFO] Frames: {total_frames}")

        # Writer for output video
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            print(f"[INFO] Output: {output_path}")

        all_results: list[FrameResult] = []
        frame_idx = 0

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                hands = self._detect_hands(frame_rgb, frame_w, frame_h)

                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    hands=hands
                    )
                all_results.append(frame_result)

                annotated = self._draw_annotations(frame_bgr.copy(), frame_result)

                if frame_idx % 30 == 0:
                    print(
                        f"[Frame {frame_idx:5d}/{total_frames}] "
                        f"Hands: {len(hands)}"
                    )

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("HOI Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] User interrupt")
                        break

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.mp_hands.close()

        print(f"[INFO] Done analyzing. analyzed frames: {len(all_results)}")
        return all_results

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering / annotazioni
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:

        for hand in result.hands:
            self._draw_hand(frame, hand)

        cv2.putText(frame,
                    f"Hands: {len(result.hands)} ",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_hand(self, frame: np.ndarray, hand: HandData) -> None:
        """draw 21 keypoints and hand bbox."""
        h, w = frame.shape[:2]
        color_map = {
            "Right": (0, 255, 127),   
            "Left":  (255, 165, 0),
        }
        color = color_map.get(hand.handedness, (200, 200, 200))

        connections = MP_HANDS.HAND_CONNECTIONS
        for conn in connections:
            pt1 = hand.keypoints[conn[0]].astype(int)
            pt2 = hand.keypoints[conn[1]].astype(int)
            cv2.line(frame, tuple(pt1), tuple(pt2), color, 2, cv2.LINE_AA)

        # Keypoints
        for idx, kp in enumerate(hand.keypoints):
            x, y = int(kp[0]), int(kp[1])
            r = 5 if idx in FINGERTIP_INDICES else 3
            cv2.circle(frame, (x, y), r, color, -1)
            cv2.circle(frame, (x, y), r, (255, 255, 255), 1) 

        # Bounding box
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, hand.handedness, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Centroid
        c = hand.centroid.astype(int)
        cv2.drawMarker(frame, tuple(c), color,
                       cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

# ──────────────────────────────────────────────────────────────────────────────
# post-processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_hand_velocity(results: list[FrameResult], fps: float) -> list[np.ndarray]:
    """
    compute hand velocity
    """
    velocities = [np.zeros((len(results[0].hands), 2), dtype=np.float32)]
    for t in range(1, len(results)):
        curr_hands = results[t].hands
        prev_hands = results[t - 1].hands
        dt = (results[t].timestamp_ms - results[t - 1].timestamp_ms) / 1000.0
        if dt <= 0:
            dt = 1.0 / fps
        n = min(len(curr_hands), len(prev_hands))
        vel = np.zeros((len(curr_hands), 2), dtype=np.float32)
        for i in range(n):
            vel[i] = (curr_hands[i].centroid - prev_hands[i].centroid) / dt
        velocities.append(vel)
    return velocities


def print_frame_summary(result: FrameResult) -> None:
    """textual recap of the frame."""
    print(f"\n── Frame {result.frame_idx} (t={result.timestamp_ms:.0f} ms) ──")

    if result.hands:
        for i, hand in enumerate(result.hands):
            print(f"  Hand [{i}] {hand.handedness}:")
            print(f"    Centroid:    ({hand.centroid[0]:.1f}, {hand.centroid[1]:.1f})")
            print(f"    Wrist (kp0):  ({hand.wrist[0]:.1f}, {hand.wrist[1]:.1f})")
            print(f"    Spread R(t):  {hand.hand_spread():.2f} px")
            print(f"    Bbox:         {hand.bbox}")
    else:
        print("  No hand detected.")

    if result.objects:
        for obj in result.objects:
            print(f"  Object [{obj.track_id}] '{obj.label}' "
                  f"conf={obj.confidence:.2f} "
                  f"center=({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})")
    else:
        print("  No object detected.")

    dists = result.hand_object_distances()
    if dists:
        print("  Hand-object distance [pixel]:")
        for (h_idx, t_id), d in dists.items():
            print(f"    Hand[{h_idx}] ↔ Obj[{t_id}]: {d:.1f} px")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video Hand-Object Interaction Tracker "
                    "(MediaPipe + YOLOv8 + ByteTrack)"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="input video path(.mp4)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="outout annotated video path (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not show video during video processing"
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip N frames"
    )
    parser.add_argument(
        "--max-hands", type=int, default=1,
        help="Max number of hands to detect (default: 2)"
    )
    parser.add_argument(
        "--mp-detect-conf", type=float, default=0.5,
        help="Confidence threshold MediaPipe localizer (default: 0.5)"
    )
    parser.add_argument(
        "--mp-track-conf", type=float, default=0.5,
        help="Confidence threshold tracking MediaPipe (default: 0.5)"
    )

    parser.add_argument(
        "--verbose", action="store_true",
        help="Print frame details"
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # Validazione input
    if not Path(args.input).exists():
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)
    
    tracker = VideoHOITracker(
        mp_max_num_hands=args.max_hands,
        mp_min_detection_confidence=args.mp_detect_conf,
        mp_min_tracking_confidence=args.mp_track_conf
    )
    
    results = tracker.process_video(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display,
        skip_frames=args.skip,
    )
    
    # ── Post-processing opzionale ─────────────────────────────────────────────
    if args.verbose:
        for r in results[:5]:  # stampa solo i primi 5 frame per brevità
            print_frame_summary(r)
    
    # Calcolo velocità (disponibile per uso successivo, es. cost function)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if len(results) > 1:
        hand_vel = compute_hand_velocity(results, fps)
        print(f"\n[INFO] Velocity not computed for {len(hand_vel)} frame.")
    
    print("\n[DONE] finished.")
