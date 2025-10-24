import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string
import signal
import sys

# === CONFIG ===
video_path_1 = '/home/oasees/AV_3_6/mast_video_reduced.mp4'  ##  DJI_0097.MP4
video_path_2 = '/home/oasees/AV_3_6/DJI_0094.MP4'     # DJI_0098_CUT.mp4
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'
#rtmp_url = 'rtmp://127.0.0.1/live/mystream'
inference_interval = 1
latency_threshold = 0.09
grayscale_mode = False  # Set True to start in grayscale mode

# === START FFmpeg STREAMING (RTSP + RTMP) ===
ffmpeg_cmd = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1', '-i', video_path_1,
    '-vf', 'fps=10,scale=1280:720',
    '-c:v', 'libx264',
    '-b:v', '2000k',
    '-g', '30', '-keyint_min', '30',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-crf', '23',
    '-f', 'tee',
    '-map', '0:v',
    f"[f=rtsp:rtsp_transport=tcp]{rtsp_url}" ## |[f=flv]{rtmp_url}
]

print("?? Launching FFmpeg for RTSP + RTMP...")
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

# === LOAD YOLOv5 MODEL ===
print("?? Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt', device='cpu') # 
model.conf = 0.3
model.eval()

# === OPEN BOTH VIDEOS ===
cap1 = cv2.VideoCapture(video_path_1)
cap2 = cv2.VideoCapture(video_path_2)

if not cap1.isOpened() or not cap2.isOpened():
    print("? Failed to open one or both video sources.")
    ffmpeg_proc.terminate()
    sys.exit(1)

# === FLASK APP ===
app = Flask(__name__)

def gen_frames():
    frame_count = 0
    last_rendered = None

    while True:
        t0 = time.time()
        #ret1, frame1 = cap1.read()
        #ret2, frame2 = cap2.read()

        #if not ret1 or frame1 is None or not ret2 or frame2 is None:
        #    print("?? One of the streams ended or dropped a frame.")
        #    break
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or frame1 is None:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()

        if not ret2 or frame2 is None:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("? Unable to rewind one of the videos.")
            break
        
        
        
        grab_latency = time.time() - t0
        if grab_latency > latency_threshold:
            print(f"?? Frame dropped due to latency ({grab_latency:.3f}s > {latency_threshold}s)")
            continue

        frame_count += 1
        merged = cv2.hconcat([cv2.resize(frame1, (640, 720)), cv2.resize(frame2, (640, 720))])
        
        #merged = cv2.hconcat([cv2, cv2])  
        
        if grayscale_mode:
            gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
            merged = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif frame_count % inference_interval == 0:
            print(f"?? Running inference on frame {frame_count}...")
            results = model(merged)
            merged = results.render()[0]

        ret, buffer = cv2.imencode('.jpg', merged)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>YOLOv5 Dual Video Inference</title></head>
        <body>
            <h2>Dual Stream YOLOv5 Inference ({{ mode }})</h2>
            <img src="{{ url_for('video_feed') }}" width="1280">
        </body>
        </html>
    """, mode="Grayscale" if grayscale_mode else "Inference")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === CLEANUP ===
def cleanup(signum=None, frame=None):
    print("?? Shutting down...")
    if cap1.isOpened():
        cap1.release()
    if cap2.isOpened():
        cap2.release()
    if ffmpeg_proc:
        ffmpeg_proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# === START SERVER ===
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        cleanup()
