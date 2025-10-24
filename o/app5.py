from flask import Flask, Response, render_template_string
import torch
import cv2
import subprocess
import time

# ---------- CONFIGURATION ----------
ffmpeg_input = '/home/oasees/AV_3_6/DJI_0097.MP4'
stream_name = 'mystream'
rtmp_url = f"rtmp://127.0.0.1/live/{stream_name}"
# -----------------------------------

# Start FFmpeg to push video to RTMP stream
ffmpeg_command = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',
    '-i', ffmpeg_input,
    '-r', '5',
    '-vf', 'scale=640:360',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '28',
    '-tune', 'zerolatency',
    '-f', 'flv',
    rtmp_url
]

print(f"Streaming video to {rtmp_url}...")
ffmpeg_proc = subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)  # Wait for stream to become active

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt')
model.conf = 0.3
model.eval()

# Open RTMP stream for inference
cap = cv2.VideoCapture(rtmp_url)

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue  # Skip failed frames

        # Run inference
        results = model(frame)
        rendered_frame = results.render()[0]

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', rendered_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <head><title>YOLOv5 RTMP Stream</title></head>
            <body>
                <h2>YOLOv5 Inference (RTMP, Edge Mode)</h2>
                <img src="{{ url_for('video_feed') }}">
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
        print("Shutting down FFmpeg...")
        ffmpeg_proc.terminate()
