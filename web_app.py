 
import os
import time
from typing import List, Tuple, Dict
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from ultralytics import YOLO

from utils import (
    to_centroids,
    match_and_movements,
    overlay_heatmap,
    draw_boxes,
    draw_hud,
    Throttler,
)

# Optional: MediaPipe Pose for fall/fight heuristics
try:
    import mediapipe as mp  # type: ignore
    mp_pose = mp.solutions.pose
    _POSE_OK = True
except Exception:
    mp_pose = None  # type: ignore
    _POSE_OK = False

# Optional: DeepFace for facial expressions
try:
    from deepface import DeepFace  # type: ignore
    _DEEPFACE_OK = True
except Exception:
    DeepFace = None  # type: ignore
    _DEEPFACE_OK = False

app = Flask(__name__)

# Config
SOURCE = 0  # laptop webcam (can be URL string too)
GRID_ROWS = 6
GRID_COLS = 8
PANIC_THRESHOLD = 60.0
MAX_MATCH_DIST = 80.0
ALPHA = 0.30
RESIZE_WIDTH = 640  # set 0 to disable
MODEL_WEIGHTS = "yolov8n.pt"

# Accuracy/perf knobs
DEVICE = "auto"     # auto|cpu|cuda|0|1
IMGSZ = 640          # 512/640 typical
FP16 = False         # True if CUDA GPU and want FP16
LOW_LATENCY = True   # request MJPG + small buffer for webcams
CAMERA_FPS = 30
CAMERA_WIDTH = 0     # request width (0 = leave default)
CAMERA_HEIGHT = 0    # request height (0 = leave default)

# Alerts thresholds
CROWD_THRESHOLD = 10  # high crowd when people >= this
CELL_DWELL_SEC = 30   # density must persist to alert
CELL_COUNT_THRESHOLD = 6  # people per cell threshold (proxy for 5 ppl/sqm)

# Fall/Fight heuristics
POSE_SAMPLE_RATE = 5  # analyze pose every N frames
FALL_MIN_TILT = 55.0  # torso tilt deg to consider prone
FALL_STATIONARY_SEC = 2.0
FIGHT_HAND_SPEED = 30.0  # px/frame threshold
FIGHT_NEAR_DIST = 120.0  # px between persons

# Emotion analysis
EMOTION_SAMPLE_RATE = 5  # analyze every N frames
EMOTION_NEGATIVE_THRESHOLD = 0.5  # ratio of negative emotions
EMOTION_MAX_FACES = 2

# Globals
model = None
cap = None
prev_centroids: List[Tuple[int, int]] = []
fps = 0.0
last_time = time.time()
sound_throttle = Throttler(min_interval=3.0)
sms_throttle = Throttler(min_interval=20.0)
HEATMAP_ENABLED = True
latest_stats = {
    "crowd_count": 0,
    "max_movement": 0.0,
    "panic": False,
    "high_crowd": False,
    "anomaly": False,
    "density_ratio": 0.0,
    "neg_emotion_ratio": 0.0,
    "faces_analyzed": 0,
    "fps": 0.0,
    "ts": time.time(),
}
# Rolling history for charts (last N points)
HIST_MAX = 300
history = deque(maxlen=HIST_MAX)  # each item: (ts, people, movement, fps, density, panic, high_crowd, anomaly)

# Incident log (recent)
INC_MAX = 100
incidents = deque(maxlen=INC_MAX)  # dicts with ts, type, level, msg

# Density dwell tracking per cell
_cell_start: Dict[Tuple[int, int], float] = {}

# Twilio SMS (optional)
def send_alert_sms(message: str) -> None:
    try:
        from twilio.rest import Client  # type: ignore
        sid = os.getenv('TWILIO_ACCOUNT_SID')
        token = os.getenv('TWILIO_AUTH_TOKEN')
        to = os.getenv('ALERT_SMS_TO')
        sender = os.getenv('TWILIO_FROM')
        if not all([sid, token, to, sender]):
            return
        client = Client(sid, token)
        client.messages.create(body=message, from_=sender, to=to)
    except Exception:
        # silent fail to avoid breaking the loop
        pass

try:
    from playsound import playsound  # type: ignore
    sound_enabled = True
except Exception:
    playsound = None  # type: ignore
    sound_enabled = False


def init_camera_and_model():
    global model, cap
    if model is None:
        model = YOLO(MODEL_WEIGHTS)
    if cap is None:
        # Use DirectShow on Windows for local cams to reduce latency
        if isinstance(SOURCE, int):
            cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {SOURCE}")
        # Camera tuning
        try:
            if CAMERA_WIDTH > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            if CAMERA_HEIGHT > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            if CAMERA_FPS > 0:
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            if LOW_LATENCY:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass


def gen_frames():
    global prev_centroids, fps, last_time, latest_stats
    init_camera_and_model()
    pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False) if _POSE_OK else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if RESIZE_WIDTH and frame.shape[1] > 0:
            H, W = frame.shape[:2]
            new_w = RESIZE_WIDTH
            new_h = int(H * (new_w / float(W)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Inference with perf knobs
        run_device = DEVICE
        if DEVICE == "auto":
            try:
                import torch
                run_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                run_device = "cpu"

        use_half = bool(FP16 and str(run_device).startswith("cuda"))
        results = model(frame, imgsz=IMGSZ, device=run_device, half=use_half, verbose=False)
        r = results[0]
        boxes = r.boxes

        if boxes is None or boxes.xyxy is None or boxes.cls is None or len(boxes) == 0:
            person_boxes_xyxy = np.empty((0, 4), dtype=int)
        else:
            try:
                import torch
                person_mask = (boxes.cls == 0) if isinstance(boxes.cls, torch.Tensor) else (boxes.cls == 0)
            except Exception:
                person_mask = np.ones((len(boxes.xyxy),), dtype=bool)
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            if isinstance(person_mask, np.ndarray):
                person_boxes_xyxy = xyxy[person_mask]
            else:
                person_boxes_xyxy = xyxy[np.array(person_mask.cpu().numpy(), dtype=bool)]

        centroids = to_centroids(person_boxes_xyxy)
        movements = match_and_movements(prev_centroids, centroids, max_dist=MAX_MATCH_DIST)
        panic = bool(len(movements) > 0 and max(movements) > PANIC_THRESHOLD)

        # High crowd vs anomaly flags (anomaly from motion and/or emotions)
        high_crowd = len(person_boxes_xyxy) >= CROWD_THRESHOLD
        anomaly = bool(panic)

        # Facial expression analysis (sampled)
        negative_ratio = 0.0
        analyzed_faces = 0
        if _DEEPFACE_OK and (len(person_boxes_xyxy) > 0) and ((len(history) % max(1, EMOTION_SAMPLE_RATE)) == 0):
            negative_emotions = {"angry", "fear", "sad", "disgust"}
            max_faces = min(max(1, EMOTION_MAX_FACES), len(person_boxes_xyxy))
            neg = 0
            for i in range(max_faces):
                x1, y1, x2, y2 = person_boxes_xyxy[i]
                h = max(0, y2 - y1)
                face_y2 = y1 + int(0.45 * h)
                fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), max(0, x2), max(0, face_y2)
                if fx2 <= fx1 or fy2 <= fy1:
                    continue
                face_crop = frame[fy1:fy2, fx1:fx2]
                try:
                    res = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False)
                    if isinstance(res, list) and len(res) > 0:
                        res = res[0]
                    dominant = res.get("dominant_emotion")
                    if dominant and str(dominant).lower() in negative_emotions:
                        neg += 1
                    analyzed_faces += 1
                except Exception:
                    pass
            if analyzed_faces > 0:
                negative_ratio = neg / float(analyzed_faces)
                if negative_ratio >= EMOTION_NEGATIVE_THRESHOLD:
                    anomaly = True

        # Build grid occupancy for dwell alerts
        Hh, Ww = frame.shape[:2]
        cell_h = max(1, Hh // GRID_ROWS)
        cell_w = max(1, Ww // GRID_COLS)
        grid_counts = {}
        for (cx, cy) in to_centroids(person_boxes_xyxy):
            r = min(int(cy // cell_h), GRID_ROWS - 1)
            c = min(int(cx // cell_w), GRID_COLS - 1)
            grid_counts[(r, c)] = grid_counts.get((r, c), 0) + 1

        # Update dwell timers and raise density incidents
        now = time.time()
        for rc, count in grid_counts.items():
            if count >= CELL_COUNT_THRESHOLD:
                if rc not in _cell_start:
                    _cell_start[rc] = now
                elif (now - _cell_start[rc]) >= CELL_DWELL_SEC:
                    r, c = rc
                    msg = f"High density sustained in cell {r}-{c}"
                    incidents.append({"ts": now, "type": "density", "level": "🟠 High", "msg": msg})
                    send_alert_sms(f"[CrowdSense] {msg}")
                    _cell_start[rc] = now + 1e9  # prevent spamming; reset later when clears
            else:
                if rc in _cell_start:
                    del _cell_start[rc]

        # Pose-based fall/fight heuristics (sampled)
        fall_detected = False
        fight_detected = False
        wrists_prev: List[Tuple[int, int]] = []
        wrists_curr: List[Tuple[int, int]] = []
        torsos_tilt: List[float] = []

        if pose is not None and (len(person_boxes_xyxy) > 0) and ((len(history) % POSE_SAMPLE_RATE) == 0):
            for (x1, y1, x2, y2) in person_boxes_xyxy[:4]:  # limit for perf
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = pose.process(crop_rgb)
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks.landmark
                def pt(i):
                    return (int(x1 + lm[i].x * (x2 - x1)), int(y1 + lm[i].y * (y2 - y1)))
                # Torso tilt: vector shoulder->hip
                ls, lh = pt(11), pt(23)
                rs, rh = pt(12), pt(24)
                sx = (ls[0] + rs[0]) * 0.5; sy = (ls[1] + rs[1]) * 0.5
                hx = (lh[0] + rh[0]) * 0.5; hy = (lh[1] + rh[1]) * 0.5
                vx, vy = (hx - sx), (hy - sy)
                angle = abs(np.degrees(np.arctan2(vy, max(1e-3, vx))))
                tilt = min(angle, 180 - angle)  # 0 upright, 90 horizontal
                torsos_tilt.append(tilt)
                # Wrists positions
                lw, rw = pt(15), pt(16)
                wrists_curr.extend([lw, rw])
            # Simple fall heuristic: strong horizontal torso and low vertical position
            if torsos_tilt and max(torsos_tilt) >= FALL_MIN_TILT:
                fall_detected = True
            # Simple fight heuristic: fast hands and proximity
            # Estimate hand speed vs previous frame
            if 'wrists_cache' not in globals():
                globals()['wrists_cache'] = []
            wrists_prev = globals()['wrists_cache']
            speed = 0.0
            for i in range(min(len(wrists_prev), len(wrists_curr))):
                px, py = wrists_prev[i]
                cx, cy = wrists_curr[i]
                speed = max(speed, ((cx - px)**2 + (cy - py)**2)**0.5)
            globals()['wrists_cache'] = wrists_curr
            if speed >= FIGHT_HAND_SPEED and len(centroids) >= 2:
                # Check proximity
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        dx = centroids[i][0] - centroids[j][0]
                        dy = centroids[i][1] - centroids[j][1]
                        if (dx*dx + dy*dy) ** 0.5 <= FIGHT_NEAR_DIST:
                            fight_detected = True
                            break
                    if fight_detected:
                        break

        # Raise incidents
        if fall_detected:
            msg = "Possible fall detected"
            incidents.append({"ts": now, "type": "fall", "level": "🔴 Critical", "msg": msg})
            send_alert_sms(f"[CrowdSense] {msg}")
            anomaly = True
        if fight_detected:
            msg = "Possible fight detected"
            incidents.append({"ts": now, "type": "fight", "level": "🔴 Critical", "msg": msg})
            send_alert_sms(f"[CrowdSense] {msg}")
            anomaly = True

        # Panic/anomaly SMS alert (throttled)
        if (panic or anomaly) and sms_throttle.can_run():
            try:
                summary = []
                if panic:
                    summary.append("panic")
                if fall_detected:
                    summary.append("fall")
                if fight_detected:
                    summary.append("fight")
                if analyzed_faces:
                    summary.append(f"negEmo={negative_ratio:.2f}")
                send_alert_sms("[CrowdSense] Alert: " + ", ".join(summary) if summary else "[CrowdSense] Alert triggered")
            finally:
                sms_throttle.ran()

        # Optionally overlay heatmap
        if HEATMAP_ENABLED:
            frame = overlay_heatmap(frame, centroids, GRID_ROWS, GRID_COLS, alpha=ALPHA)
        draw_boxes(frame, person_boxes_xyxy, color=(0, 255, 0), thickness=2)

        # FPS calc
        curr_time = time.time()
        dt = curr_time - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        last_time = curr_time

        draw_hud(frame, crowd_count=len(person_boxes_xyxy), movements=movements, panic=panic, fps=fps)

        # Update latest stats for dashboard polling
        density_ratio = 0.0
        total_cells = max(1, GRID_ROWS * GRID_COLS)
        density_ratio = min(1.0, float(len(person_boxes_xyxy)) / float(total_cells * 2))  # rough proxy
        latest_stats = {
            "crowd_count": int(len(person_boxes_xyxy)),
            "max_movement": float(max(movements) if movements else 0.0),
            "panic": bool(panic),
            "high_crowd": bool(high_crowd),
            "anomaly": bool(anomaly),
            "density_ratio": float(density_ratio),
            "neg_emotion_ratio": float(negative_ratio),
            "faces_analyzed": int(analyzed_faces),
            "fps": float(fps),
            "ts": time.time(),
        }

        # Append to history
        history.append((latest_stats["ts"], latest_stats["crowd_count"], latest_stats["max_movement"], latest_stats["fps"], latest_stats["density_ratio"], latest_stats["panic"], latest_stats["high_crowd"], latest_stats["anomaly"]))

        if panic and sound_enabled and playsound is not None and sound_throttle.can_run():
            try:
                playsound('alert.mp3', block=False)
            except Exception:
                pass
            sound_throttle.ran()

        # Encode as JPEG
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        jpg = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

        prev_centroids = centroids


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify({**latest_stats, "incidents": list(incidents)})


@app.route('/history')
def history_endpoint():
    # Return arrays for charting
    ts, people, movement, fps_list, density, panic, high_crowd, anomaly = ([] for _ in range(8))
    for item in list(history):
        t, p, m, f, d, pa, hc, an = item
        ts.append(t)
        people.append(p)
        movement.append(m)
        fps_list.append(f)
        density.append(d)
        panic.append(pa)
        high_crowd.append(hc)
        anomaly.append(an)
    return jsonify({
        "ts": ts,
        "people": people,
        "movement": movement,
        "fps": fps_list,
        "density": density,
        "panic": panic,
        "high_crowd": high_crowd,
        "anomaly": anomaly,
    })


@app.route('/weather')
def weather():
    try:
        import requests  # local import
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        if not lat or not lon:
            return jsonify({"ok": False, "error": "lat/lon required"}), 400
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return jsonify({"ok": False, "error": "server missing OPENWEATHER_API_KEY"}), 500
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=5)
        data = r.json()
        temp = data.get('main', {}).get('temp')
        cond = (data.get('weather') or [{}])[0].get('main')
        return jsonify({"ok": True, "temp": temp, "cond": cond})
    except Exception:
        return jsonify({"ok": False}), 500


@app.route('/set_heatmap', methods=['POST'])
def set_heatmap():
    global HEATMAP_ENABLED
    try:
        data = request.get_json(force=True, silent=True) or {}
        enabled = bool(data.get('enabled'))
        HEATMAP_ENABLED = enabled
        return jsonify({"ok": True, "heatmap": HEATMAP_ENABLED})
    except Exception:
        return jsonify({"ok": False}), 400


if __name__ == '__main__':
    # Start Flask server
    # Open http://127.0.0.1:5000 in your browser
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
