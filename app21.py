import subprocess
import time
import cv2
import torch
from flask import Flask, Response, render_template_string, jsonify
import signal
import sys
import os
import csv
from datetime import datetime

# === CONFIG ===
video_list = [
    '/home/oasees/AV_3_6/mast_video_reduced.mp4',
    '/home/oasees/AV_3_6/DJI_0094.MP4'
]
selected_video_index = 0
rtsp_url = 'rtsp://127.0.0.1:8554/mystream'
inference_interval = 1
latency_threshold = 0.09
grayscale_mode = False
merge_toggle = True
use_dual_stream = True  # Toggle to False to use stream selection

allowed_class_names = ['corrosion', 'moderate corrosion', 'rust', 'severe corrosion']       #['severe corrosion', 'moderate corrosion', 'corrosion']  # ? Change this list as needed


# === FFmpeg STREAM (for demo) ===
ffmpeg_proc = None
if use_dual_stream:
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',
        '-stream_loop', '-1', '-i', video_list[0],
        '-vf', 'fps=10,scale=1280:720',
        '-c:v', 'libx264',
        '-b:v', '2000k',
        '-g', '30', '-keyint_min', '30',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-crf', '23',
        '-f', 'tee',
        '-map', '0:v',
        f"[f=rtsp:rtsp_transport=tcp]{rtsp_url}"
    ]
    print("??? Launching FFmpeg for RTSP...")
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

# === LOAD MODEL ===
print("?? Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/home/oasees/AV_3_6/Weights_yolov5_2_7_25_v7_45epochs/best.pt',      #
                       device='cpu')
model.conf = 0.2
model.iou = 0.5
model.agnostic = True
model.multi_label = True
model.eval()

# === FLASK APP ===
app = Flask(__name__)
cap1 = cv2.VideoCapture(video_list[selected_video_index])
cap2 = cv2.VideoCapture(video_list[1]) if use_dual_stream else None
log_file_path = "detections_log.csv"

# === Initialize Log File ===


# === Initialize Log File ===
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])



#def log_detections(results):
#    now = datetime.now().isoformat()
#    for *xyxy, conf, cls in results.xyxy[0]:
#        class_id = int(cls)
#        class_name = model.names[class_id] if class_id in model.names else "unknown"
#        with open(log_file_path, 'a', newline='') as csvfile:
#            writer = csv.writer(csvfile)
#            writer.writerow([now, class_id, class_name, float(conf), *map(int, xyxy)])


def log_detections(results):
    now = datetime.now().isoformat()
    for *xyxy, conf, cls in results.xyxy[0]:
        class_id = int(cls)
        class_name = model.names.get(class_id, "unknown")

        if class_name not in allowed_class_names:
            continue  # ? Skip this detection

        with open(log_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([now, class_id, class_name, float(conf), *map(int, xyxy)])




def gen_frames():
    global cap1, cap2
    frame_count = 0
    while True:
        t0 = time.time()
        ret1, frame1 = cap1.read()
        ret2, frame2 = (True, None)
        if use_dual_stream and cap2:
            ret2, frame2 = cap2.read()

        if not ret1 or frame1 is None:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()

        if use_dual_stream and (not ret2 or frame2 is None):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()

        if not ret1 or (use_dual_stream and not ret2):
            print("?? Unable to rewind.")
            break

        grab_latency = time.time() - t0
        if grab_latency > latency_threshold:
            print(f"?? Frame dropped: {grab_latency:.3f}s")
            continue

        frame_count += 1
        merged = frame1

        if use_dual_stream and merge_toggle:
            merged = cv2.hconcat([
                cv2.resize(frame1, (640, 720)),
                cv2.resize(frame2, (640, 720))
            ])
        else:
            merged = cv2.resize(frame1, (1280, 720))

        if grayscale_mode:
            merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
            merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)

        elif frame_count % inference_interval == 0:
            print(f"?? Inference on frame {frame_count}")
            resized_input = cv2.resize(merged, (960, 960))
            results = model(resized_input)
            merged = results.render()[0]
            log_detections(results)

        ret, buffer = cv2.imencode('.jpg', merged)
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
            <h2>YOLOv5 Detection - {{ mode }}</h2>
            <img src="{{ url_for('video_feed') }}" width="1280"><br><br>
            <p>Use: /select_video/0 or /select_video/1 to switch stream</p>
        </body>
        </html>
    """, mode="Grayscale" if grayscale_mode else "Inference")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_video/<int:index>')
def select_video(index):
    global cap1, selected_video_index
    if index < 0 or index >= len(video_list):
        return jsonify({"error": "Invalid video index."}), 400

    print(f"?? Switching to video index {index}")
    selected_video_index = index
    if cap1.isOpened():
        cap1.release()
    cap1 = cv2.VideoCapture(video_list[selected_video_index])
    return jsonify({"message": f"Switched to video {video_list[selected_video_index]}"})

@app.route('/change-mode')
def change_mode():
    global merge_toggle
    merge_toggle = not merge_toggle
    return jsonify("Mode changed.")

# === CLEANUP ===
def cleanup(signum=None, frame=None):
    print("?? Cleaning up...")
    if cap1.isOpened():
        cap1.release()
    if cap2 and cap2.isOpened():
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
