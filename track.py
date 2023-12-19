import threading
import cv2
from ultralytics import YOLO
import queue


def run_tracker_in_thread(filename, model, file_index, frame_queue):
    try:
        video = cv2.VideoCapture(filename)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            res_plotted = results[0].plot()
            frame_queue.put((file_index, res_plotted))
        video.release()
    except Exception as e:
        print(f"Error in thread {file_index}: {e}")


frame_queue = queue.Queue()
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n-seg.pt")
model3 = YOLO("yolov8n.pt")
# VIDEO_FILE_1 = "/Users/agustin/Workspace/personal/yolo8/asd.mp4"
VIDEO_FILE_1 = 0
VIDEI_FILE_2 = 1

tracker_thread1 = threading.Thread(
    target=run_tracker_in_thread,
    args=(VIDEO_FILE_1, model1, 1, frame_queue),
    daemon=True,
)
tracker_thread2 = threading.Thread(
    target=run_tracker_in_thread,
    args=(VIDEI_FILE_2, model3, 2, frame_queue),
    daemon=True,
)
tracker_thread1.start()
tracker_thread2.start()

try:
    while True:
        file_index, frame = frame_queue.get(block=True)  # Change to block=True
        cv2.imshow(f"Tracking_Stream_{file_index}", frame)
        if cv2.waitKey(1) == ord("q"):
            break
except queue.Empty:
    print("Queue is empty")
finally:
    cv2.destroyAllWindows()
    tracker_thread1.join()
    tracker_thread2.join()
