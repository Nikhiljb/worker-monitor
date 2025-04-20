import cv2
from ultralytics import YOLO
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock

# Threaded frame capture
class ThreadedVideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(f"{src}?rtsp_transport=udp", cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.stopped = False
        self.frame = None
        self.lock = Lock()
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.stream.grab():
                ret, frame = self.stream.retrieve()
                if ret:
                    with self.lock:
                        self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# Load YOLOv8n model
model = YOLO("D:/start_worker_activity_monitor/yolov8n.pt")

# For threaded video input (change path or rtsp URL as needed)
video_path = "D:/start_worker_activity_monitor/office work.mp4"
# If it's a local file, use standard capture; otherwise, you can use ThreadedVideoStream for RTSP
use_threaded = False

if use_threaded:
    cap = ThreadedVideoStream(video_path)
else:
    cap = cv2.VideoCapture(video_path)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Create log file
log_dir = Path("D:/start_worker_activity_monitor/")
log_dir.mkdir(parents=True, exist_ok=True)
log_filename = datetime.now().strftime("worker_log_yolo_%Y%m%d_%H%M%S.txt")
log_file_path = log_dir / log_filename
log_file = open(log_file_path, "w")

# Parameters
motion_threshold = 500
required_motion_duration = 0  # seconds
motion_start_time = None
current_status = "Idle"
last_logged_status = None

while True:
    if use_threaded:
        frame = cap.read()
        if frame is None:
            continue
    else:
        ret, frame = cap.read()
        if not ret:
            break

    fgmask = fgbg.apply(frame)
    results = model(frame, imgsz=480, conf=0.35, iou=0.5)[0]
    detected_motion = False

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi_mask = fgmask[y1:y2, x1:x2]
            motion_score = cv2.countNonZero(roi_mask)

            if motion_score > motion_threshold:
                detected_motion = True
                if motion_start_time is None:
                    motion_start_time = time.time()
            else:
                motion_start_time = None

            if motion_start_time is not None:
                elapsed_time = time.time() - motion_start_time
                if elapsed_time >= required_motion_duration:
                    current_status = "Working"
                else:
                    current_status = "Idle"
            else:
                current_status = "Idle"

            color = (0, 255, 0) if current_status == "Working" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, current_status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if current_status != last_logged_status:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{timestamp} - {current_status}\n")
                print(f"{timestamp} - {current_status}")
                last_logged_status = current_status

    cv2.imshow("Worker Activity Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if use_threaded:
    cap.stop()
else:
    cap.release()

log_file.close()
cv2.destroyAllWindows()
