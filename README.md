# Face Detection & Student Tracking

A real-time face detection and student tracking web app built with **Flask** and **OpenCV** on a **Raspberry Pi 4**. The Pi camera feed is processed with an OpenCV DNN face detector, detected faces are labelled (Student 1, Student 2, ...) and saved, and the annotated video is streamed live to the browser as an MJPEG feed with a small control dashboard.

## Features

- Live face detection from the Pi camera using **OpenCV DNN** (res10 SSD)
- Real-time video streamed to the browser (MJPEG, no page refresh)
- Detected faces labelled and cropped face images saved to `captures/`
- Browser keyboard controls: **D** to start the stream, **S** to stop
- Live dashboard: face count, students saved, uptime, activity log

## Tech Stack

- **Python 3**
- **Flask** — web server, MJPEG streaming, control endpoints
- **OpenCV (DNN module)** — face detection
- **Picamera2** — Raspberry Pi camera capture
- **Raspberry Pi 4** + Pi Camera V2

## Project Structure

```
face-detection-flask/
├── app.py                 # Flask server + detection loop
├── templates/
│   └── index.html         # Dashboard UI
├── models/                # DNN model files (auto-downloaded on first run)
├── captures/              # Saved face images (git-ignored)
├── requirements.txt
└── .gitignore
```

## Setup

> These steps are for Raspberry Pi OS.

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-detection-flask.git
   cd face-detection-flask
   ```

2. Install the camera library (system package) and create a venv that can see it:
   ```bash
   sudo apt install -y python3-picamera2 libcap-dev
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   ```

3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running

```bash
python app.py
```

The face detection model files download automatically on the first run. Then open a browser and go to:

```
http://<your-pi-ip>:5000
```

Press **D** to start detecting and **S** to stop.

## Notes

- This project originally used MediaPipe, but switched to **OpenCV DNN** because `mediapipe-rpi4` only supports up to Python 3.9 (the Pi was on Python 3.13).
- The Pi Camera V2 outputs native RGB888 via Picamera2, so frames are kept in RGB throughout and only converted to BGR at the JPEG encode/save step to avoid a color cast.
- The `captures/` folder is git-ignored on purpose — it holds people's face images and should not go in a public repo.

## Author

Jey — Woosong University (Endicott College program)
