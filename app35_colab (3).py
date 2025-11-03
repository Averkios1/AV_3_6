# ==========================
# app35_colab.py
# ==========================
import torch
import cv2
import numpy as np
import pandas as pd
import os
from google.colab.patches import cv2_imshow
from datetime import datetime
from ultralytics import YOLO

# ========= CONFIG ==========
ROOT = '/content/AV_3_6'

# Choose inference mode: 1 or 2 videos
INFER_MODE = 2   # <-- set to 1 or 2

VIDEO_PATH_1 = os.path.join(ROOT, 'mast_video_reduced.mp4')
VIDEO_PATH_2 = os.path.join(ROOT, 'final_combined_video4.mp4')
WEIGHTS_PATH = os.path.join(ROOT, 'Weights_yolov5_24_9_25_v14_with_183images_45epochs/best.pt')
OUTPUT_LOG = '/content/detections_log.csv'
OUTPUT_VIDEO = '/content/output_detection.mp4'

# ========== LOAD MODEL ==========
print(f"🔹 Loading YOLOv5 model from: {WEIGHTS_PATH}")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS_PATH, trust_repo=True)
#model = YOLO(WEIGHTS_PATH)
model.conf = 0.25
model.iou = 0.5
print("✅ Model loaded successfully.")

# ========== INIT VIDEO(S) ==========
cap1 = cv2.VideoCapture(VIDEO_PATH_1)
cap2 = cv2.VideoCapture(VIDEO_PATH_2) if INFER_MODE == 2 else None

if not cap1.isOpened() or (INFER_MODE == 2 and not cap2.isOpened()):
    raise Exception("❌ Could not open one or both videos. Check paths!")

fps = int(cap1.get(cv2.CAP_PROP_FPS)) or 25
frame_width = 1280
frame_height = 720

out = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (frame_width, frame_height)
)

frame_count = 0
detections = []

# ========== PROCESS LOOP ==========
print(f"🚀 Starting inference in {'dual' if INFER_MODE==2 else 'single'}-video mode...")
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = (cap2.read() if INFER_MODE == 2 else (False, None))

    if not ret1 or (INFER_MODE == 2 and not ret2):
        print("End of video reached or read error.")
        break

    if INFER_MODE == 2:
        # Apply ROIs safely
        roi1 = frame1[400:1120, 400:1600] if frame1.shape[0] >= 1120 and frame1.shape[1] >= 1600 else frame1
        roi2 = frame2[0:720, 700:1200] if frame2.shape[1] >= 1200 else frame2
        roi1_resized = cv2.resize(roi1, (800, 720))
        roi2_resized = cv2.resize(roi2, (480, 720))
        merged = cv2.hconcat([roi1_resized, roi2_resized])
    else:
        roi1 = frame1[400:1120, 400:1600] if frame1.shape[0] >= 1120 and frame1.shape[1] >= 1600 else frame1
        merged = cv2.resize(roi1, (1280, 720))

    # YOLO inference
    results = model(merged)
    detections_df = results.pandas().xyxy[0]
    
    # Log detections
    for _, row in detections_df.iterrows():
        detections.append({
            "frame": frame_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "class": row["name"],
            "confidence": float(row["confidence"]),
            "x1": float(row["xmin"]),
            "y1": float(row["ymin"]),
            "x2": float(row["xmax"]),
            "y2": float(row["ymax"])
        })

    # Annotate and save frame
    rendered = results.render()[0]
    rendered = rendered.copy()
    cv2.putText(rendered, f"Frame: {frame_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(rendered)

    # Display every 10th frame
    if frame_count % 10 == 0:
        cv2_imshow(rendered)
        print(f"Processed frame {frame_count} — Detections: {len(detections_df)}")

    frame_count += 1

cap1.release()
if cap2: cap2.release()
out.release()
cv2.destroyAllWindows()

# ========== SAVE DETECTIONS ==========
if detections:
    df = pd.DataFrame(detections)
    df.to_csv(OUTPUT_LOG, index=False)
    print(f"✅ Detections saved to {OUTPUT_LOG}")
else:
    print("⚠️ No detections logged.")

print(f"🎥 Output video saved to: {OUTPUT_VIDEO}")
