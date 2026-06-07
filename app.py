"""
Face Detection & Student Tracking System
=========================================
Detector : OpenCV DNN (res10_300x300_ssd Caffe model, auto-downloads on first run)
Stream   : Flask MJPEG live feed (no page refresh)
Controls : D = start stream, S = stop stream (from the browser keyboard)
Saves    : ./captures/student_1.jpg, student_2.jpg ...
Hardware : Raspberry Pi 4 + Pi Camera V2 (Picamera2)
"""

import cv2
import numpy as np
import threading
import os
import time
import urllib.request
import logging
from flask import Flask, Response, render_template, jsonify

from picamera2 import Picamera2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR   = os.path.join(BASE_DIR, "captures")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
FRAME_W       = 640
FRAME_H       = 480
STREAM_FPS    = 20
SAVE_INTERVAL = 3.0        # seconds between saves
CONF_THRESH   = 0.6        # face detection confidence

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Model (auto-downloads on first run) ───────────────────────
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
PROTO_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_models():
    if not os.path.exists(PROTO_PATH):
        log.info("Downloading deploy.prototxt ...")
        urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)
    if not os.path.exists(MODEL_PATH):
        log.info("Downloading caffemodel (~10 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("Models ready.")


download_models()
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

# ── Shared state ──────────────────────────────────────────────
app = Flask(__name__)

_latest_frame = None
_frame_lock   = threading.Lock()
_streaming    = False

_stats = {
    "faces_now": 0,
    "students_saved": 0,
    "streaming": False,
    "started_at": time.time(),
}
_stats_lock = threading.Lock()

_last_save_time = 0.0
_student_count  = 0


def _make_paused_frame():
    """A static placeholder shown when the stream is stopped."""
    img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    cv2.putText(img, "STREAM PAUSED", (FRAME_W // 2 - 130, FRAME_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.putText(img, "press D to start", (FRAME_W // 2 - 110, FRAME_H // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    ok, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


_paused_frame = _make_paused_frame()


# ── Detection loop (runs in a background thread) ──────────────
def detection_loop():
    global _latest_frame, _last_save_time, _student_count

    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(1.0)  # camera warm-up
    log.info("Camera started.")

    interval = 1.0 / STREAM_FPS

    while True:
        t0 = time.time()

        if not _streaming:
            time.sleep(0.1)
            continue

        # Pi Camera V2 via Picamera2 outputs native RGB888 -> keep RGB throughout
        frame = picam.capture_array()

        # Detection blob in RGB order, no channel swap
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (123.0, 177.0, 104.0),
            swapRB=False, crop=False
        )
        net.setInput(blob)
        detections = net.forward()

        h, w = frame.shape[:2]
        faces_now = 0
        now = time.time()
        do_save = (now - _last_save_time) >= SAVE_INTERVAL

        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < CONF_THRESH:
                continue
            faces_now += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            label = f"Student {faces_now}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if do_save:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    _student_count += 1
                    fname = os.path.join(CAPTURE_DIR, f"student_{_student_count}.jpg")
                    # RGB -> BGR only at write time
                    cv2.imwrite(fname, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        if do_save and faces_now > 0:
            _last_save_time = now
            with _stats_lock:
                _stats["students_saved"] = _student_count

        with _stats_lock:
            _stats["faces_now"] = faces_now

        # Encode display frame: RGB -> BGR only at encode time
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

        time.sleep(max(0, interval - (time.time() - t0)))


# ── MJPEG generator ───────────────────────────────────────────
def _gen_frames():
    while True:
        if not _streaming:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + _paused_frame + b"\r\n"
            )
            time.sleep(0.5)
            continue

        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1.0 / STREAM_FPS)


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/control/<action>", methods=["POST"])
def control(action):
    global _streaming
    if action == "start":
        _streaming = True
        log.info("Stream STARTED via keypress")
    elif action == "stop":
        _streaming = False
        log.info("Stream STOPPED via keypress")
    with _stats_lock:
        _stats["streaming"] = _streaming
    return jsonify({"streaming": _streaming})


@app.route("/stats")
def stats():
    with _stats_lock:
        s = dict(_stats)
    s["uptime"] = int(time.time() - s.pop("started_at"))
    return jsonify(s)


# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    log.info("Flask -> http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
