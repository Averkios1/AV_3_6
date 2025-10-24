#from flask import Flask, Response, render_template_string
#import torch
#import cv2


#import os



from flask import Flask, Response, render_template_string
import torch
import cv2
import numpy as np
import os
import subprocess
import time


# Configuration
ffmpeg_input = '/home/oasees/AV_3_6/DJI_0097.MP4'  # Local video to stream
rtsp_stream_name = 'mystream'
raspberry_ip = '127.0.0.1'  # or actual IP if streaming to another device
rtsp_url = f"rtsp://{raspberry_ip}:8554/{rtsp_stream_name}"

# Start ffmpeg process to stream video to MediaMTX RTSP server
ffmpeg_command = [
    'ffmpeg',
    '-re',
    '-stream_loop', '-1',  # Loop the video indefinitely
    '-i', ffmpeg_input,
    '-r', '5',
    #'-c:v', 'libx264',                # Video codec
    #'-preset', 'veryfast',            # Lower CPU usage
    #'-crf', '4',                     # Compression quality (lower is better)
    #'-tune', 'zerolatency',           # Optional: Low-latency
    '-f', 'rtsp',
    '-rtsp_transport', 'udp',
    rtsp_url
]

print(f"Starting ffmpeg to stream to {rtsp_url}...")
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Give FFmpeg a few seconds to start streaming
time.sleep(5)



#   ffmpeg -re -stream_loop -1 -i /home/oasees/AV_3_6/mast_video_reduced.mp4   -c:v copy -f rtsp rtsp://127.0.0.1:8554/mystream

# Load YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # or yolov5m/l/x
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='Weights_yolov5_27_5_25/best.pt', source='local')

repo = os.path.abspath("/home/oasees/AV_3_6/yolov5")
#model = torch.hub.load(repo, 'custom', path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt', source='local')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/oasees/AV_3_6/Weights_yolov5_28_5_25/best.pt')
model.conf = 0.30

model.eval()

# Video file path
#video_path = '/home/oasees/AV_3_6/mast_video_reduced.mp4'
#video_path = '/home/oasees/AV_3_6/DJI_0097.MP4'


# Define preprocessing
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame / 255.0  # Normalize
    
    



#rtsp_url = "rtsp://127.0.0.1:8554/mystream" 



cap = cv2.VideoCapture(rtsp_url)

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue
        original_frame = frame.copy()  # Keep original for rendering
        
        # Preprocessing
        processed = preprocess_frame(frame)
        processed = (processed * 255).astype(np.uint8)  # Rescale to 0-255 if needed
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)  # Convert back to BGR

        # Inference
        results = model(frame)
        #results = model(processed)
        rendered_frame = results.render()[0]

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', rendered_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <head><title>YOLOv5 Stream</title></head>
            <body>
                <h1>YOLOv5 Inference Streaming</h1>
                <img src="{{ url_for('video_feed') }}">
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
        print("Shutting down ffmpeg...")
        ffmpeg_process.terminate()
