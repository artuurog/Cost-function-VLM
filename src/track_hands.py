"""
Video Hand-Object Interaction Tracker
======================================
Usage:
    python track_anything.py --input video.mp4 --output output.mp4
    python track_anything.py --input video.mp4 --no-display
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
        """Coordinate dei 5 fingertip, shape (5, 2)."""
        return self.keypoints[FINGERTIP_INDICES]

    def hand_spread(self) -> float:
        """
        R(t): distanza media di ogni keypoint dal polso.
        Eq. 6 nell'articolo: misura dell'apertura della mano.
        """
        dists = np.linalg.norm(self.keypoints[1:] - self.wrist, axis=1)
        return float(np.mean(dists))


# class ObjectData:
#     """Contiene i dati di un oggetto rilevato e tracciato in un frame."""

#     def __init__(self, track_id: int, label: str, confidence: float, bbox: tuple):
#         """
#         Args:
#             track_id:   ID univoco del tracker (ByteTrack)
#             label:      Etichetta COCO (es. "bottle", "cup")
#             confidence: Confidenza del rilevamento [0, 1]
#             bbox:       (x1, y1, x2, y2) bounding box in pixel
#         """
#         self.track_id = track_id
#         self.label = label
#         self.confidence = confidence
#         self.bbox = bbox

#     @property
#     def centroid(self) -> np.ndarray:
#         """
#         po_i(t): posizione del centroide dell'oggetto i al frame t.
#         Usato nelle equazioni 1-11 dell'articolo.
#         """
#         x1, y1, x2, y2 = self.bbox
#         return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


class FrameResult:
    """Risultati dell'analisi di un singolo frame."""

    def __init__(self, frame_idx: int, timestamp_ms: float,
                 hands: list[HandData]):
        self.frame_idx = frame_idx
        self.timestamp_ms = timestamp_ms
        self.hands = hands      # Lista di HandData

    # def hand_object_distances(self) -> dict:
    #     """
    #     Calcola d_i(t) = ||p_h(t) - po_i(t)|| per ogni coppia mano-oggetto.
    #     Eq. 1 nell'articolo.

    #     Returns:
    #         dict con chiavi (hand_idx, obj_track_id) e valori float (distanza pixel)
    #     """
    #     distances = {}
    #     for h_idx, hand in enumerate(self.hands):
    #         ph = hand.centroid
    #         for obj in self.objects:
    #             poi = obj.centroid
    #             dist = float(np.linalg.norm(ph - poi))
    #             distances[(h_idx, obj.track_id)] = dist
    #     return distances

    # def min_fingertip_distances(self) -> dict:
    #     """
    #     d_min_i(t): distanza minima tra i fingertip e l'oggetto i.
    #     Usato nella Eq. 7 per calcolare il peso di prossimità w_d(t).

    #     Returns:
    #         dict con chiavi (hand_idx, obj_track_id) e valori float
    #     """
    #     distances = {}
    #     for h_idx, hand in enumerate(self.hands):
    #         fingertips = hand.fingertips  # (5, 2)
    #         for obj in self.objects:
    #             poi = obj.centroid
    #             dists_to_obj = np.linalg.norm(fingertips - poi, axis=1)
    #             distances[(h_idx, obj.track_id)] = float(np.min(dists_to_obj))
    #     return distances


# ──────────────────────────────────────────────────────────────────────────────
# Tracker principale
# ──────────────────────────────────────────────────────────────────────────────

class VideoHOITracker:
    """
    Tracker Hand-Object Interaction su video MP4.

    Pipeline:
    1. Per ogni frame: rilevamento mani con MediaPipe (21 keypoints)
    2. Per ogni frame: rilevamento + tracking oggetti con YOLOv8 + ByteTrack
    3. Calcolo delle distanze mano-oggetto (base per il cost function)
    """

    def __init__(
        self,
        mp_max_num_hands: int = 2,
        mp_min_detection_confidence: float = 0.5,
        mp_min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            mp_max_num_hands:              Numero massimo di mani da rilevare
            mp_min_detection_confidence:   Soglia confidenza rilevamento MediaPipe
            mp_min_tracking_confidence:    Soglia confidenza tracking MediaPipe
        """
        # MediaPipe Hands
        self.mp_hands = MP_HANDS.Hands(
            static_image_mode=False,
            max_num_hands=mp_max_num_hands,
            min_detection_confidence=mp_min_detection_confidence,
            min_tracking_confidence=mp_min_tracking_confidence,
        )

        

    def _detect_hands(self, frame_rgb: np.ndarray, frame_w: int, frame_h: int) -> list[HandData]:
        """
        Rileva le mani nel frame RGB con MediaPipe.

        Returns:
            Lista di HandData, una per ogni mano rilevata
        """
        results = self.mp_hands.process(frame_rgb)
        hands = []

        if not results.multi_hand_landmarks:
            return hands

        for lm_list, handedness_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            handedness = handedness_info.classification[0].label  # "Left" / "Right"

            # Estrai i 21 keypoints in coordinate pixel
            keypoints = np.array(
                [[lm.x * frame_w, lm.y * frame_h] for lm in lm_list.landmark],
                dtype=np.float32
            )  # shape: (21, 2)

            # Calcola bounding box dalla nuvola di keypoints
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            pad = 15  # pixel di padding
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

    # def _detect_objects(self, frame_bgr: np.ndarray) -> list[ObjectData]:
    #     """
    #     Rileva e traccia gli oggetti nel frame con YOLOv8 + ByteTrack.

    #     Returns:
    #         Lista di ObjectData, una per ogni oggetto tracciato
    #     """
    #     if self.yolo is None:
    #         return []

    #     results = self.yolo.track(
    #         frame_bgr,
    #         persist=True,           # mantiene gli ID di tracking tra i frame
    #         tracker="bytetrack.yaml",
    #         conf=self.yolo_confidence,
    #         classes=self.yolo_classes,
    #         verbose=False,
    #     )

    #     objects = []
    #     if results[0].boxes is None:
    #         return objects

    #     boxes = results[0].boxes
    #     names = results[0].names

    #     for box in boxes:
    #         # Salta oggetti senza track ID (ByteTrack non ancora assegnato)
    #         if box.id is None:
    #             continue

    #         track_id = int(box.id.item())
    #         cls_id = int(box.cls.item())
    #         label = names[cls_id]
    #         conf = float(box.conf.item())
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         bbox = (int(x1), int(y1), int(x2), int(y2))

    #         objects.append(ObjectData(
    #             track_id=track_id,
    #             label=label,
    #             confidence=conf,
    #             bbox=bbox,
    #         ))

    #     return objects

    def process_video(
        self,
        input_path: str,
        output_path: str | None = None,
        display: bool = True,
        skip_frames: int = 0,
    ) -> list[FrameResult]:
        """
        Processa un video MP4 frame per frame.

        Args:
            input_path:  Percorso al video di input (.mp4)
            output_path: Percorso al video di output annotato (None = non salvare)
            display:     Mostra il video a schermo durante l'elaborazione
            skip_frames: Salta N frame tra un'analisi e l'altra (0 = analizza tutti)

        Returns:
            Lista di FrameResult, uno per ogni frame analizzato
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossibile aprire il video: {input_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Video: {input_path}")
        print(f"[INFO] Risoluzione: {frame_w}x{frame_h} @ {fps:.1f} fps")
        print(f"[INFO] Frame totali: {total_frames}")

        # Writer per il video di output
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

                # Salta frame se richiesto
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # ── 1. Rilevamento mani ───────────────────────────────────────
                hands = self._detect_hands(frame_rgb, frame_w, frame_h)

                # ── 3. Salva risultati ────────────────────────────────────────
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    hands=hands
                    )
                all_results.append(frame_result)

                # ── 4. Annotazione visiva ─────────────────────────────────────
                annotated = self._draw_annotations(frame_bgr.copy(), frame_result)

                # ── 5. Stampa info a console ──────────────────────────────────
                if frame_idx % 30 == 0:
                    print(
                        f"[Frame {frame_idx:5d}/{total_frames}] "
                        f"Mani: {len(hands)}"
                    )

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("HOI Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] Interruzione da utente (tasto Q)")
                        break

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self.mp_hands.close()

        print(f"[INFO] Elaborazione completata. Frame analizzati: {len(all_results)}")
        return all_results

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering / annotazioni
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_annotations(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Disegna mani, oggetti e distanze sul frame."""

        # Converti in RGB per MediaPipe drawing, poi torna BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # ── Mani ─────────────────────────────────────────────────────────────
        for hand in result.hands:
            self._draw_hand(frame, hand)

        # # ── Linee mano-oggetto (distanze) ─────────────────────────────────────
        # for hand in result.hands:
        #     ph = hand.centroid.astype(int)
        #     for obj in result.objects:
        #         poi = obj.centroid.astype(int)
        #         dist = np.linalg.norm(hand.centroid - obj.centroid)
        #         # Colore: verde = vicino, rosso = lontano (soglia 150px)
        #         color = (0, 200, 0) if dist < 150 else (0, 100, 200)
        #         cv2.line(frame, tuple(ph), tuple(poi), color, 1, cv2.LINE_AA)
        #         mid = ((ph[0] + poi[0]) // 2, (ph[1] + poi[1]) // 2)
        #         cv2.putText(frame, f"{dist:.0f}px", mid,
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # ── HUD info ──────────────────────────────────────────────────────────
        cv2.putText(frame,
                    f"Mani: {len(result.hands)} ",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _draw_hand(self, frame: np.ndarray, hand: HandData) -> None:
        """Disegna i 21 keypoints e il bounding box della mano."""
        h, w = frame.shape[:2]
        color_map = {
            "Right": (0, 255, 127),   # verde acqua
            "Left":  (255, 165, 0),   # arancione
        }
        color = color_map.get(hand.handedness, (200, 200, 200))

        # Connessioni MediaPipe (skeleton della mano)
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
            cv2.circle(frame, (x, y), r, (255, 255, 255), 1)  # bordo bianco

        # Bounding box
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, hand.handedness, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Centroide
        c = hand.centroid.astype(int)
        cv2.drawMarker(frame, tuple(c), color,
                       cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    # def _draw_object(self, frame: np.ndarray) -> None:
    #     """Disegna il bounding box e il centroide dell'oggetto tracciato."""
    #     PALETTE = [
    #         (86, 180, 233), (230, 159, 0), (204, 121, 167),
    #         (0, 158, 115), (213, 94, 0), (0, 114, 178),
    #     ]
    #     color = PALETTE[obj.track_id % len(PALETTE)]
    #     x1, y1, x2, y2 = obj.bbox

    #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #     label_text = f"[{obj.track_id}] {obj.label} {obj.confidence:.2f}"
    #     cv2.putText(frame, label_text, (x1, y2 + 16),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

    #     # Centroide po_i(t)
    #     c = obj.centroid.astype(int)
    #     cv2.circle(frame, tuple(c), 5, color, -1)
    #     cv2.circle(frame, tuple(c), 5, (255, 255, 255), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Funzioni di utilità post-processing
# ──────────────────────────────────────────────────────────────────────────────

def compute_hand_velocity(results: list[FrameResult], fps: float) -> list[np.ndarray]:
    """
    Calcola v_h(t): velocità della mano in pixel/secondo.
    Sezione III.A dell'articolo.

    Returns:
        Lista di array (N_mani, 2) con velocità per ogni frame.
        Il primo frame ha velocità zero.
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


# def compute_hand_object_relative_velocity(
#     results: list[FrameResult], fps: float
# ) -> list[dict]:
#     """
#     Calcola v_oi(t): velocità relativa mano-oggetto in pixel/secondo.
#     Derivata di d_i(t) rispetto al tempo (Sezione III.A).

#     Returns:
#         Lista di dict {(hand_idx, track_id): velocità_relativa float}
#     """
#     rel_velocities = [{}]
#     for t in range(1, len(results)):
#         curr = results[t]
#         prev = results[t - 1]
#         dt = (curr.timestamp_ms - prev.timestamp_ms) / 1000.0
#         if dt <= 0:
#             dt = 1.0 / fps

#         curr_dists = curr.hand_object_distances()
#         prev_dists = prev.hand_object_distances()

#         rv = {}
#         for key in curr_dists:
#             if key in prev_dists:
#                 rv[key] = abs(curr_dists[key] - prev_dists[key]) / dt
#         rel_velocities.append(rv)

#     return rel_velocities


def print_frame_summary(result: FrameResult) -> None:
    """Stampa un riepilogo testuale del frame."""
    print(f"\n── Frame {result.frame_idx} (t={result.timestamp_ms:.0f} ms) ──")

    if result.hands:
        for i, hand in enumerate(result.hands):
            print(f"  Mano [{i}] {hand.handedness}:")
            print(f"    Centroide:    ({hand.centroid[0]:.1f}, {hand.centroid[1]:.1f})")
            print(f"    Polso (kp0):  ({hand.wrist[0]:.1f}, {hand.wrist[1]:.1f})")
            print(f"    Spread R(t):  {hand.hand_spread():.2f} px")
            print(f"    Bbox:         {hand.bbox}")
    else:
        print("  Nessuna mano rilevata.")

    if result.objects:
        for obj in result.objects:
            print(f"  Oggetto [{obj.track_id}] '{obj.label}' "
                  f"conf={obj.confidence:.2f} "
                  f"centro=({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})")
    else:
        print("  Nessun oggetto rilevato.")

    dists = result.hand_object_distances()
    if dists:
        print("  Distanze mano-oggetto [pixel]:")
        for (h_idx, t_id), d in dists.items():
            print(f"    Mano[{h_idx}] ↔ Obj[{t_id}]: {d:.1f} px")


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
        help="Percorso al video di input (.mp4)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Percorso al video di output annotato (opzionale)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Non mostrare il video durante l'elaborazione"
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Salta N frame tra un'analisi e l'altra (default: 0)"
    )
    parser.add_argument(
        "--max-hands", type=int, default=2,
        help="Numero massimo di mani da rilevare (default: 2)"
    )
    parser.add_argument(
        "--mp-detect-conf", type=float, default=0.5,
        help="Soglia confidenza rilevamento MediaPipe (default: 0.5)"
    )
    parser.add_argument(
        "--mp-track-conf", type=float, default=0.5,
        help="Soglia confidenza tracking MediaPipe (default: 0.5)"
    )
    parser.add_argument(
        "--yolo-model", default="yolov8n.pt",
        help="Modello YOLO da usare (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=0.4,
        help="Soglia confidenza YOLO (default: 0.4)"
    )
    parser.add_argument(
        "--yolo-classes", nargs="+", type=int, default=None,
        help="ID classi COCO da rilevare (default: tutte). "
             "Es: --yolo-classes 39 41 67 (bottiglia, tazza, telefono)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Stampa dettagli per ogni frame"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validazione input
    if not Path(args.input).exists():
        print(f"[ERROR] File non trovato: {args.input}")
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
        # rel_vel = compute_hand_object_relative_velocity(results, fps)
        print(f"\n[INFO] Velocità mano calcolata per {len(hand_vel)} frame.")
        # print(f"[INFO] Velocità relativa mano-oggetto calcolata per {len(rel_vel)} frame.")

    print("\n[DONE] Elaborazione terminata.")


if __name__ == "__main__":
    main()