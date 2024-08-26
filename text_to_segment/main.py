import sieve
import cv2
import shutil
import os


# get the first frame of a video as a sieve.File
def get_first_frame(video: sieve.File):
    video_path = video.path

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('first_frame.png', frame)
    else:
        raise Exception("Failed to read the video")

    frame = sieve.File(path='first_frame.png')

    return frame


def get_object_bbox(video: sieve.File, object_name: str):
    yolo = sieve.function.get('sieve/yolov8')

    frame = get_first_frame(video)

    response = yolo.run(
        file=frame,
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]

    return box


@sieve.function(
    name="text-to-segment",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ]
)
def segment(video: sieve.File, object_name: str):
    sam = sieve.function.get("sieve-internal/sam2")

    box = get_object_bbox(video, object_name)

    sam_prompt = {
        "frame_index": 0,
        "object_id": 1,
        "box": [box['x1'],box['y1'],box['x2'],box['y2']]
    }

    debug, response = sam.run(
        file=video,
        prompts=[sam_prompt],
        model_type="tiny",
        pixel_confidences=True,
        debug_masks=True,
        bbox_tracking=True
    )

    return debug, response



if __name__ == "__main__":
    video_path = "trolley.mp4"

    video = sieve.File(path=video_path)
    # box = get_object_bbox(video, "trolley")
    debug, response = segment(video, "trolley")

    shutil.move(debug.path, "output.mp4")

    breakpoint()


