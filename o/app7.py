import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string

# === CONFIG ===
video_path = '/home/oasees/AV_3_6/DJI_0097.MP4'  # Path to your local video
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'      # RTSP stream output

# === START FFMPEG STREAM WITH HIGH QUALITY SETTINGS ===
ffmpeg_cmd = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',
    '-i', video_path,
    '-r', '15',                  # Higher FPS for smoother video
    '-vf', 'scale=1280x720',   # Full HD resolution
    '-c:v', 'libx264',
    '-b:v', '3000k',             # 3 Mbps bitrate for better quality
    '-preset', 'slow',           # Better compression efficiency
    '-tune', 'zerolatency',      # Low latency for streaming
    '-crf', '18',                # Near visually lossless quality
    '-f', 'rtsp',
    '-rtsp_transport', 'udp',
    rtsp_url
]

print(f"?? Starting FFmpeg stream to {rtsp_url}...")
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)  # Give FFmpeg a few seconds to start streaming

# === LOAD YOLOv5 MODEL ===
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt')
model.conf = 0.3
model.eval()

# === OPEN RTSP STREAM ===
cap = cv2.VideoCapture(rtsp_url)

# === FLASK APP ===
app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("?? Stream read error, retrying...")
            time.sleep(0.5)
            continue

        # Perform inference
        results = model(frame)
        rendered = results.render()[0]

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', rendered)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
        <head><title>YOLOv5 RTSP Stream</title></head>
        <body>
            <h2>YOLOv5 Detection Stream from RTSP (1080p)</h2>
            <img src="{{ url_for('video_feed') }}" width="960">
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        print("?? Stopping FFmpeg and releasing resources...")
        ffmpeg_proc.terminate()
        cap.release()
