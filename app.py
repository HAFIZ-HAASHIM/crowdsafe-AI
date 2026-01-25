import argparse
import time
from typing import List, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    to_centroids,
    match_and_movements,
    overlay_heatmap,
    draw_boxes,
    draw_hud,
    Throttler,
)

# Optional: facial expression analysis via DeepFace
try:
    from deepface import DeepFace  # type: ignore
    _DEEPFACE_AVAILABLE = True
except Exception:
    DeepFace = None  # type: ignore
    _DEEPFACE_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdSense MVP — Crowd density and movement monitoring")
    parser.add_argument("--source", type=str, default="0", help="Video source: camera index (e.g., 0) or stream URL/file path")
    parser.add_argument("--grid_rows", type=int, default=5, help="Grid rows for density heatmap")
    parser.add_argument("--grid_cols", type=int, default=5, help="Grid cols for density heatmap")
    parser.add_argument("--panic_threshold", type=float, default=60.0, help="Pixel movement threshold for panic detection")
    parser.add_argument("--max_match_dist", type=float, default=80.0, help="Max distance to match centroids between frames")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for heatmap overlay (0-1)")
    parser.add_argument("--resize_width", type=int, default=0, help="Resize frame to this width (keep aspect). 0 = no resize")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics YOLOv8 model weights")
    # Performance/accuracy knobs
    parser.add_argument("--device", type=str, default="auto", help="Inference device: auto|cpu|cuda|0|1 ...")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO (e.g., 512/640)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (requires CUDA)")
    # Camera capture tuning
    parser.add_argument("--camera_width", type=int, default=0, help="Request camera width (0=leave default)")
    parser.add_argument("--camera_height", type=int, default=0, help="Request camera height (0=leave default)")
    parser.add_argument("--camera_fps", type=int, default=30, help="Request camera FPS")
    parser.add_argument("--low_latency", action="store_true", help="Set low-latency capture (MJPG + small buffer)")
    # New: explicit crowd density and emotion anomaly thresholds
    parser.add_argument("--crowd_threshold", type=int, default=10, help="People count above which to raise HIGH CROWD alert")
    parser.add_argument(
        "--emotion_negative_threshold",
        type=float,
        default=0.5,
        help="Proportion of negative emotions among analyzed faces to flag ANOMALY (0-1)",
    )
    parser.add_argument(
        "--emotion_sample_rate",
        type=int,
        default=5,
        help="Analyze emotions every N frames to maintain FPS. Set 1 to analyze every frame",
    )
    parser.add_argument("--disable_emotion", action="store_true", help="Disable DeepFace emotion analysis")
    parser.add_argument("--max_faces", type=int, default=2, help="Max faces to analyze per sample")
    return parser.parse_args()


def str_to_source(src: str) -> Union[int, str]:
    # If src is an integer string, treat as camera index
    try:
        return int(src)
    except ValueError:
        return src


def main():
    args = parse_args()

    source = str_to_source(args.source)
    # Use DirectShow for local cameras on Windows to reduce latency
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    # Camera tuning
    try:
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        if args.low_latency:
            # Request MJPG stream and small buffer if supported
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("[INFO] Loading model:", args.model)
    model = YOLO(args.model)

    prev_centroids: List[Tuple[int, int]] = []
    fps = 0.0
    last_time = time.time()

    # Optional sound throttle (no audio file included by default)
    try:
        from playsound import playsound  # type: ignore
        sound_enabled = True
    except Exception:
        playsound = None  # type: ignore
        sound_enabled = False
    sound_throttle = Throttler(min_interval=3.0)

    frame_idx = 0
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed or stream ended.")
            break

        # Optional resize to boost FPS
        if args.resize_width and frame.shape[1] > 0:
            H, W = frame.shape[:2]
            new_w = args.resize_width
            new_h = int(H * (new_w / float(W)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Inference (pass performance knobs)
        # Resolve device automatically if requested
        run_device = args.device
        if args.device == "auto":
            try:
                import torch
                run_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                run_device = "cpu"

        use_half = False
        if args.fp16 and run_device in ("cuda", "0", "1", "2"):
            use_half = True

        results = model(
            frame,
            imgsz=args.imgsz,
            device=run_device,
            half=use_half,
            verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        # Filter class==0 (person)
        person_boxes_xyxy: np.ndarray
        if boxes is None or boxes.xyxy is None or boxes.cls is None or len(boxes) == 0:
            person_boxes_xyxy = np.empty((0, 4), dtype=int)
        else:
            try:
                import torch  # local import to avoid hard dep if not needed
                person_mask = (boxes.cls == 0) if isinstance(boxes.cls, torch.Tensor) else (boxes.cls == 0)
            except Exception:
                # Fallback: assume all boxes are persons (not ideal but safe fallback for MVP)
                person_mask = np.ones((len(boxes.xyxy),), dtype=bool)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            if isinstance(person_mask, np.ndarray):
                person_boxes_xyxy = xyxy[person_mask]
            else:
                # torch tensor mask
                person_boxes_xyxy = xyxy[np.array(person_mask.cpu().numpy(), dtype=bool)]

        # Centroids and movements
        centroids = to_centroids(person_boxes_xyxy)
        movements = match_and_movements(prev_centroids, centroids, max_dist=args.max_match_dist)

        # Panic detection (anomaly component 1)
        panic = bool(len(movements) > 0 and max(movements) > args.panic_threshold)

        # Facial expression analysis (anomaly component 2)
        negative_emotions = {"angry", "fear", "sad", "disgust"}
        analyzed_faces = 0
        negative_count = 0
        emotions_info: List[Tuple[str, float]] = []  # (dominant_emotion, confidence)

        do_emotion = (not args.disable_emotion) and _DEEPFACE_AVAILABLE and (frame_idx % max(1, args.emotion_sample_rate) == 0)
        if do_emotion and len(person_boxes_xyxy) > 0:
            # Limit number of faces to analyze per cycle for performance
            max_faces = min(max(1, args.max_faces), len(person_boxes_xyxy))
            # Take up to first N persons; you could also sample
            for i in range(max_faces):
                x1, y1, x2, y2 = person_boxes_xyxy[i]
                # Heuristic face crop: upper part of the person box
                h = max(0, y2 - y1)
                face_y2 = y1 + int(0.45 * h)
                fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), max(0, x2), max(0, face_y2)
                if fx2 <= fx1 or fy2 <= fy1:
                    continue
                face_crop = frame[fy1:fy2, fx1:fx2]
                try:
                    # Analyze emotions; DeepFace returns a dict per image
                    result = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)
                    # DeepFace may return list or dict depending on version
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    dominant = result.get("dominant_emotion")
                    scores = result.get("emotion", {})
                    conf = float(scores.get(dominant, 0.0)) if isinstance(scores, dict) else 0.0
                    if dominant:
                        emotions_info.append((str(dominant), conf))
                        analyzed_faces += 1
                        if str(dominant).lower() in negative_emotions:
                            negative_count += 1
                except Exception:
                    # Skip analysis errors to keep real-time performance
                    pass

        negative_ratio = (negative_count / analyzed_faces) if analyzed_faces > 0 else 0.0

        # Alert conditions
        high_crowd = len(person_boxes_xyxy) >= args.crowd_threshold
        anomaly = panic or (negative_ratio >= args.emotion_negative_threshold)

        # Overlays
        frame = overlay_heatmap(frame, centroids, args.grid_rows, args.grid_cols, alpha=args.alpha)
        draw_boxes(frame, person_boxes_xyxy, color=(0, 255, 0), thickness=2)

        # Distinct alert banners
        banner_y = 30
        if high_crowd:
            # Red banner for HIGH CROWD
            cv2.rectangle(frame, (10, banner_y - 22), (10 + 420, banner_y + 8), (0, 0, 255), thickness=-1)
            cv2.putText(
                frame,
                f"ALERT: HIGH CROWD DENSITY (count={len(person_boxes_xyxy)})",
                (16, banner_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            banner_y += 32
        if anomaly:
            # Orange banner for ANOMALOUS ACTIVITY
            cv2.rectangle(frame, (10, banner_y - 22), (10 + 520, banner_y + 8), (0, 140, 255), thickness=-1)
            msg = "ALERT: ANOMALOUS ACTIVITY"
            if analyzed_faces > 0:
                msg += f" | neg_emotion_ratio={negative_ratio:.2f}"
            cv2.putText(
                frame,
                msg,
                (16, banner_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # FPS
        curr_time = time.time()
        dt = curr_time - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_time = curr_time

        # HUD (existing)
        draw_hud(frame, crowd_count=len(person_boxes_xyxy), movements=movements, panic=panic, fps=fps)

        # Optional sound alert (trigger on either alert type)
        if (anomaly or high_crowd) and sound_enabled and playsound is not None and sound_throttle.can_run():
            try:
                # Provide your own 'alert.mp3' in project root if desired
                playsound('alert.mp3', block=False)
            except Exception:
                pass
            sound_throttle.ran()

        cv2.imshow("CrowdSense Dashboard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        prev_centroids = centroids
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
