import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string
import signal
import sys
import os

# === CONFIG ===
video_path = '/home/oasees/AV_3_6/DJI_0097.MP4'  # Path to your local video
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'      # RTSP stream output

# === START FFMPEG STREAM WITH FRAME SKIP (5 FPS) AND SCALING ===
ffmpeg_cmd = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',
    '-i', video_path,
    '-vf', 'fps=3,scale=1280:720',  # Drop frames to 5 FPS and scale to 720p
    '-c:v', 'libx264',              # Use h264_v4l2m2m or h264_omx for HW acceleration  libx264  h264_omx
    '-b:v', '2000k',
    '-g', '30', '-keyint_min', '30',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-crf', '23',
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    rtsp_url
]

print(f"\U0001F3AC Launching FFmpeg stream to {rtsp_url} with 5 FPS and scaling...")
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Wait for the RTSP server to buffer
time.sleep(5)

# === LOAD YOLOv5 MODEL ===
print("\U0001F916 Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt')
model.conf = 0.3
model.eval()

# === OPEN RTSP STREAM ===
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("\u2753 Failed to open RTSP stream.")
    ffmpeg_proc.terminate()
    sys.exit(1)

# === FLASK APP ===
app = Flask(__name__)

def gen_frames():
    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.1)
                continue

            # Inference
            results = model(frame)
            rendered = results.render()[0]

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', rendered)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"\u2753 Error during frame processing: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>YOLOv5 RTSP Stream</title></head>
        <body>
            <h2>YOLOv5 Detection Stream (1280x720, 5 FPS)</h2>
            <img src="{{ url_for('video_feed') }}" width="960">
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === CLEANUP ON EXIT ===
def cleanup(signum=None, frame=None):
    print("\U0001F6A9 Shutting down...")
    if cap.isOpened():
        cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# === START FLASK SERVER ===
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        cleanup()
