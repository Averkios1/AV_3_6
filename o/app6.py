import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string

# === CONFIG ===
video_path = '/home/oasees/AV_3_6/DJI_0097.MP4'  # Path to your local video
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'      # RTSP stream output

# === START FFMPEG STREAM ===
ffmpeg_cmd = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',
    '-i', video_path,
    '-r', '15',  # Higher FPS than 5
    '-vf', 'scale=1280x720',  # Better resolution
    '-c:v', 'libx264',
    '-b:v', '1000k',  # Target bitrate
    '-preset', 'medium',
    '-tune', 'zerolatency',
    '-crf', '23',  # Better visual quality
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    rtsp_url
]

print(f"?? Starting FFmpeg stream to {rtsp_url}...")
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)  # Let stream stabilize

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
            print("?? Stream read error.")
            time.sleep(0.5)
            continue

        # Inference
        results = model(frame)
        rendered = results.render()[0]

        # Encode for browser
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
        <head><title>YOLOv5 Inference Stream</title></head>
        <body>
            <h2>Streaming YOLOv5 Detection from RTSP</h2>
            <img src="{{ url_for('video_feed') }}" width="720">
        </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        print("?? Stopping FFmpeg and releasing resources...")
        ffmpeg_proc.terminate()
        cap.release()
