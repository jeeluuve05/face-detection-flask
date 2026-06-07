# Face Detection & Student Tracking

A real-time face detection and student tracking web app built with **Flask** and **OpenCV** on a **Raspberry Pi 4**. The camera feed is processed with an OpenCV DNN face detector and streamed live to the browser as an MJPEG video feed.

## Features

- Live face detection from the Pi camera using OpenCV DNN
- Real-time video streamed to the browser (MJPEG)
- Student tracking on the detected faces
- Lightweight — runs on a Raspberry Pi 4

## Tech Stack

- **Python 3**
- **Flask** — web server and video streaming route
- **OpenCV (DNN module)** — face detection
- **Raspberry Pi 4** — hardware

## Hardware

- Raspberry Pi 4
- Pi Camera (or USB webcam)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-detection-flask.git
   cd face-detection-flask
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

```bash
python app.py
```

Then open your browser and go to:

```
http://<your-pi-ip>:5000
```

You'll see the live camera feed with face detection.

## Notes

- This project originally used MediaPipe, but switched to OpenCV DNN due to a Python version incompatibility on the Pi.
- The detector model files need to be present in the project folder for detection to work.

## Author

jeeluuve05 — Woosong University (Endicott College program)
