import threading
import queue
import cv2
from ultralytics import YOLO

# pylint: disable=no-member, broad-except


def run_tracker_in_thread(
    thread_filename, thread_model, thread_file_index, thread_frame_queue
):
    """
    Runs object tracking on a video source in a separate thread.

    Args:
        thread_filename (str): The path to the video file or camera index.
        thread_model (YOLO): The YOLO model object for object detection and tracking.
        thread_file_index (int): An index to uniquely identify the video source.
        thread_frame_queue (queue.Queue): The queue to put processed frames into.
    """
    try:
        video = cv2.VideoCapture(thread_filename)
        while True:
            ret, video_frame = video.read()
            if not ret:
                break
            results = thread_model.track(video_frame, persist=True)
            res_plotted = results[0].plot()
            thread_frame_queue.put((thread_file_index, res_plotted))
        video.release()
    except FileNotFoundError as e:
        print(f"File not found error in thread {thread_file_index}: {e}")
    except IOError as e:
        print(f"I/O error in thread {thread_file_index}: {e}")
    except (
        Exception
    ) as e:  # Using Exception as a general catch-all for unexpected errors
        print(f"Unexpected error in thread {thread_file_index}: {e}")


def main():
    """
    Runs object tracking on multiple video sources in separate threads.
    """
    videos_to_track = [
        "/Users/agustin/Workspace/personal/yolo8/asd.mp4",
        0,
        1,
        "/Users/agustin/Workspace/personal/yolo8/asd.mp4",
        0,
        1,
    ]

    frame_queue = queue.Queue()
    threads = []

    for index, video_file in enumerate(videos_to_track):
        model = YOLO("yolov8n.pt") if index % 2 == 0 else YOLO("yolov8n-seg.pt")
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
