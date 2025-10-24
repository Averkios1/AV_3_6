import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string
import signal
import sys

# === CONFIG ===
video_path = '/home/oasees/AV_3_6/mast_video_reduced.mp4' #  DJI_0097.MP4
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'
#rtmp_url = 'rtmp://127.0.0.1/live/mystream'
inference_interval = 1  # Run inference every N frames
latency_threshold = 0.09  # Drop frames if grab time exceeds 100ms

# === START FFMPEG STREAM (RTSP + RTMP) ===
ffmpeg_cmd = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',
    '-i', video_path,
    '-vf', 'fps=10,scale=1280:720',
    '-c:v', 'libx264',  # or 'h264_v4l2m2m' for Pi HW acceleration
    '-b:v', '2000k',
    '-g', '30', '-keyint_min', '30',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-crf', '23',
    '-f', 'tee',
    '-map', '0:v',
    f"[f=rtsp:rtsp_transport=tcp]{rtsp_url}"  # |[f=flv]{rtmp_url}
]

print("?? Launching FFmpeg for RTSP + RTMP...")
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Allow time for stream to initialize
time.sleep(5)

# === LOAD YOLOv5 MODEL ===
print("?? Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt')
model.conf = 0.3
model.eval()

# === OPEN RTSP STREAM ===
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("? Failed to open RTSP stream.")
    ffmpeg_proc.terminate()
    sys.exit(1)

# === FLASK APP ===
app = Flask(__name__)

def gen_frames():
    frame_count = 0
    last_rendered = None

    while True:
        start = time.time()
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        grab_latency = time.time() - start
        if grab_latency > latency_threshold:
            print(f"?? Frame dropped (latency {grab_latency:.3f}s > {latency_threshold}s)")
            continue

        frame_count += 1

        if frame_count % inference_interval == 0:
            inf_start = time.time()
            results = model(frame)
            last_rendered = results.render()[0]
            print(f"?? Inference frame {frame_count} (took {time.time() - inf_start:.3f}s)")

        if last_rendered is None:
            continue

        ret, buffer = cv2.imencode('.jpg', last_rendered)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head><title>YOLOv5 Stream</title></head>
    <body>
        <h2>YOLOv5 Detection (RTSP + RTMP, inference every {{ interval }} frames)</h2>
        <img src="{{ url_for('video_feed') }}" width="960">
    </body>
    </html>
    """, interval=inference_interval)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === CLEANUP ===
def cleanup(signum=None, frame=None):
    print("?? Shutting down...")
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
