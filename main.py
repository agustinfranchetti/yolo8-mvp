# pylint: disable=no-member, broad-except, missing-module-docstring

import threading
import queue
import cv2
from ultralytics import YOLO
from tracker.tracker_thread import run_tracker_in_thread


def main():
    """
    Runs object tracking on multiple video sources in separate threads.
    """
    videos_to_track = [
        "/Users/agustin/Workspace/personal/yolo8/video/asd.mp4",
        0,
        1,
    ]

    frame_queue = queue.Queue()
    threads = []

    for index, video_file in enumerate(videos_to_track):
        model = (
            YOLO("models/yolov8n.pt")
            if index % 2 == 0
            else YOLO("models/yolov8n-seg.pt")
        )
        thread = threading.Thread(
            target=run_tracker_in_thread,
            args=(video_file, model, index, frame_queue),
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    try:
        while True:
            file_index, frame = frame_queue.get(block=True)
            cv2.imshow(f"Tracking_Stream_{file_index}", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    except queue.Empty:
        print("Queue is empty")
    finally:
        cv2.destroyAllWindows()
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    main()
