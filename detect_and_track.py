# detect_and_track.py

import cv2
from ultralytics import YOLO
from deepsort.tracker import Tracker


# CONFIG

VIDEO_PATH = '15sec_input_720p.mp4'
OUTPUT_PATH = 'output.mp4'
MODEL_PATH = 'best.pt'  # YOLO checkpoint
CONFIDENCE_THRESHOLD = 0.5


# Load YOLO model + DeepSORT tracker

model = YOLO(MODEL_PATH)
print(model.names)
tracker = Tracker()


# input video

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_idx = 0


# Run detection + tracking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"CLS: {cls_id}  CONF: {conf}")

        if cls_id == 2 and conf >= CONFIDENCE_THRESHOLD:
            
            x1, y1, x2, y2 = box.xyxy[0]
            detections.append([x1.item(), y1.item(), x2.item(), y2.item()])

    print(f"[Frame {frame_idx}] Detections passed filter: {len(detections)}")
    tracker.update(detections)

    for track in tracker.tracks:
        if track.time_since_update > 0:
            continue

        x1, y1, x2, y2 = track.to_tlbr()
        track_id = track.track_id

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            2
        )
        cv2.putText(
            frame,
            f'Player {track_id}',
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"[DONE] Output saved: {OUTPUT_PATH}")
