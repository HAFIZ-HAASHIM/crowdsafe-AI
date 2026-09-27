# 🛡️ CrowdSafe AI (CrowdSense) Architecture & Vision Pipeline

CrowdSafe AI is a computer-vision powered safety telemetry system engineered to prevent crowd crushes, stampedes, and dangerous density bottlenecks in public arenas, transit hubs, and venues.

---

## 👁️ Computer Vision & Analytics Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Ingestion Stream                   │
│           (CCTV Feed · RTSP · MP4 / Video File)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Frame Extraction & Preprocessing          │
│             (OpenCV · Aspect Ratio Resizing · BGR2RGB)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Object Detection & Tracking               │
│             (YOLOv8 Head & Person Detection · DeepSORT)     │
└───────┬──────────────────────┬──────────────────────┬───────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Density Heat │       │ Flow Vector  │       │ Bottleneck   │
│ Map Analysis │       │ Estimation   │       │ Detector     │
└───────┬──────┘       └───────┬──────┘       └───────┬──────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Risk Scoring & Alert Dispatch             │
│        (Threshold Decision Logic · Twilio SMS · WebSockets) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Security Command Dashboard                │
│                 (Flask · WebSocket Streaming · Charts)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Algorithms

1. **Spatial Density Heatmaps:** Computes Gaussian kernel density estimation over bounding box centroids.
2. **Velocity Vector Tracking:** Detects sudden crowd turbulence or counter-flow anomalies preceding stampedes.
3. **Threshold-Based Early Warning:** Triggers automated audio/SMS notifications when density exceeds 4 persons/m².
