<<<<<<< HEAD
# CrowdSense — Crowd Density & Movement Monitoring (MVP)

Real-time people detection, density heatmap, and abnormal movement alerting using a single camera.

## Quickstart

1) Create and activate a virtual environment (Windows PowerShell):
```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Run with default webcam (index 0):
```
python app.py --source 0
```

4) Optional: run with IP camera or video file
```
python app.py --source http://<phone-ip>:8080/video
python app.py --source demo_videos/sample.mp4
```

## Arguments (tuning)
- `--grid_rows`, `--grid_cols`: heatmap resolution (default 5x5)
- `--panic_threshold`: px movement to trigger alert (default 60)
- `--max_match_dist`: px distance for centroid matching (default 80)
- `--alpha`: heatmap transparency (default 0.25)
- `--resize_width`: resize width to boost FPS (e.g., 640)
- `--model`: YOLOv8 weights (default `yolov8n.pt`)

## Notes
- First run will auto-download YOLOv8 weights.
- Add `alert.mp3` in project root if you want a sound alert (optional).
- Press `q` to quit.

## Next steps (post-MVP)
- Replace greedy matching with SORT/ByteTrack for stable tracking IDs.
- Zone-specific analytics and alerting.
- Persist logs (CSV/DB) and build a richer UI if needed.
=======
# crowdsafe-AI
CrowdSafe AI is an AI-powered crowd monitoring and safety system designed to prevent overcrowding and stampede incidents. It uses intelligent analysis to detect crowd density risks early and support proactive decision-making for safer public spaces and events.
>>>>>>> 2a8999d34c8c8580bc5f6f1748065ed809ce63e1
