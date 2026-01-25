# 🛡️ CrowdSafe AI (CrowdSense)

CrowdSafe AI is an AI-powered crowd monitoring and safety system designed to help prevent overcrowding and stampede incidents. Built using Python and computer vision techniques, the system analyzes crowd density in real time to support proactive safety decisions in public spaces and events.

---

## 🌍 Problem Statement

Stampede incidents often occur due to uncontrolled crowd density and delayed response. Manual monitoring is reactive and unreliable in high-density environments such as public events, religious gatherings, transport hubs, and festivals.

CrowdSafe AI addresses this challenge by enabling intelligent, data-driven crowd analysis.

---

## 🎯 Objectives

- Monitor crowd density using computer vision
- Detect overcrowding risks early
- Support preventive crowd control actions
- Improve public safety using AI-assisted insights

---

## 🚀 Features

- AI-based crowd detection using YOLO
- Real-time crowd density analysis
- Web-based interface for visualization
- Modular and scalable Python architecture
- Privacy-conscious design (no facial recognition)
- Suitable for research, demos, and hackathons

---

## 🛠️ Tech Stack

### Core Technologies
- Python
- YOLOv8 (Object Detection)
- OpenCV
- Flask (Web Application)

### Tools
- Git & GitHub
- Virtual Environment (venv)
- HTML Templates (Flask)

---

## 📂 Project Structure

```

crowdsense/
├── app.py
├── web_app.py
├── utils.py
├── requirements.txt
├── README.md
├── templates/
├── demo_videos/
├── models/          # (optional – configs only)
├── .gitignore

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/HAFIZ-HAASHIM/crowdsafe-AI.git
cd crowdsense
````

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Model Weights (IMPORTANT)

This project uses **YOLOv8** for crowd detection.

Model weights are **not included** in the repository.

### Download YOLOv8 model manually:

[https://github.com/ultralytics/assets/releases/](https://github.com/ultralytics/assets/releases/)

Place the file as:

```
yolov8n.pt
```

in the project root or `models/` directory.

---

## ▶️ Running the Application

```bash
python web_app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

---

## 🔐 Privacy & Ethics

* No facial recognition is used
* Focuses on crowd density, not individual identity
* Designed to assist authorities, not replace human judgment

---

## 📈 Future Enhancements

* Heatmap-based crowd visualization
* Live CCTV / camera integration
* Automated alert system
* IoT sensor integration
* Dashboard for authorities

---

## 👨‍💻 Author

**Muhammad Haashim**
Computer Science Student | AI & Software Developer
Founder – Codalix Agency

---

## 📄 Disclaimer

CrowdSafe AI is a research and demonstration project.
It does not guarantee prevention of incidents and should be used alongside trained personnel and established safety protocols.

---

## 📄 License

This project is intended for educational and demonstration purposes.

```

