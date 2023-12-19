# pylint: disable=no-member, broad-except, missing-module-docstring

import cv2


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
