import sieve
import cv2
import shutil
import os
import zipfile
import tempfile
import numpy as np

from blending import blend_to_background
from utils import (
    get_first_frame,
    zip_to_mp4,
    splice_audio
)


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
def segment(video: sieve.File, subject: str):
    sam = sieve.function.get("sieve/sam2")

    box = get_object_bbox(video, subject)

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




@sieve.function(
    name="background-replace",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ]
)
def background_replace(
    video: sieve.File,
    background: sieve.File,
    subject: str,
):

    _, response = segment(video, subject)

    mask_video = zip_to_mp4(response["confidences"])

    blended_vid = blend_to_background(video, mask_video, background)

    out = splice_audio(blended_vid, video)

    return out


if __name__ == "__main__":
    # video_path = "trolley.mp4"

    # video = sieve.File(path=video_path)
    # debug, response = segment(video, "trolley")

    # shutil.move(debug.path, "output.mp4")

    # breakpoint()

########################################

    # mp4 = zip_to_mp4("masks.zip")
    # shutil.move(mp4.path, "masks.mp4")

########################################

    # video_path = "trolley.mp4"
    # mask_path = "confidence.mp4"
    # bg = "galaxy.jpg"

    # output_path = blend_to_background(video_path, mask_path, bg)
    # shutil.move(output_path, "blended.mp4")

########################################

    # video_path = "trolley.mp4"
    # bg_path = "galaxy.jpg"
    # subject = "trolley"

    video_path = "musk_fixed.mp4"
    bg_path = "galaxy.jpg"
    subject = "man"

    video = sieve.File(path=video_path)
    background = sieve.File(path=bg_path)

    replaced = background_replace(video, background, subject)

    shutil.move(replaced.path, f"{video_path.split('.')[0]}final_output.mp4")




