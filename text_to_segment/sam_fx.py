import sieve
import os
import shutil
import cv2
import numpy as np

from typing import Literal

from blending import blend_to_background
from utils import splice_audio
from text_to_segment import segment
from utils import get_first_frame, resize_and_crop, resize_with_padding



def apply_shape_effect(video, mask_video, effect_mask, mask_scale=1.):


    video_reader = cv2.VideoCapture(video.path)
    mask_reader = cv2.VideoCapture(mask_video.path)

    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    effect_mask_arr = cv2.imread(effect_mask.path)
    if len(effect_mask_arr.shape) == 3:
        effect_mask_arr = cv2.cvtColor(effect_mask_arr, cv2.COLOR_BGR2GRAY)

    effect_mask_arr = resize_with_padding(effect_mask_arr, mask_scale)

    size = 2 * max(frame_width, frame_height)
    effect_mask_arr = resize_and_crop(effect_mask_arr, size, size)
    _, effect_mask_arr = cv2.threshold(effect_mask_arr, 127, 255, cv2.THRESH_BINARY)

    prev_current_center = None

    while True:
        ret_video, frame_video = video_reader.read()
        ret_mask, frame_mask = mask_reader.read()

        if not ret_video or not ret_mask:
            break

        if len(frame_mask.shape) == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

        M = cv2.moments(frame_mask, binaryImage=True)
        current_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if prev_current_center is None:
            prev_current_center = current_center

        beta = 0.2
        current_center = (int((1.-beta)*prev_current_center[0] + beta*current_center[0]), int(0.8*prev_current_center[1] + 0.2*current_center[1]))
        prev_current_center = current_center

        translation = (current_center[0] - size//2, current_center[1] - size//2)
        translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated_mask = cv2.warpAffine(effect_mask_arr, translation_matrix, (frame_mask.shape[1], frame_mask.shape[0]))

        combined_mask = cv2.bitwise_or(frame_mask, translated_mask)
        mask = combined_mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)

        new_frame = frame_video * mask + (1 - mask) * 255.

        output_writer.write(new_frame.astype(np.uint8))

    video_reader.release()
    mask_reader.release()
    output_writer.release()

    return sieve.File(path=output_path)


metadata = sieve.Metadata(
    name="Sam FX",
    description="Apply cool overlays behind a subject in a video",
    image=sieve.File(path="duck_circle.jpg")
)

@sieve.function(
    name="sam-fx",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def add_effect(
    video: sieve.File,
    subject: str,
    effect: Literal["circle", "spotlight", "frame", "retro solar"]
):
    """
    :param video: input video
    :param subject: subject of video
    :param effect: effect to overlay
    """

    print("segmenting...")
    mask_video = segment(video, subject, return_mp4=True)

    if effect == "retro solar":
        print("applying retro solar effect...")
        effect_mask = sieve.File(path="assets/rays.jpg")

        return apply_shape_effect(video, mask_video, effect_mask)

    if effect == "circle":
        print("applying circle effect...")
        effect_mask = sieve.File(path="assets/circle.jpg")

        return apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.2)

    if effect == "spotlight":
        print("applying splotlight effect...")
        effect_mask = sieve.File(path="assets/spot.jpg")

        return apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15)

    if effect == "frame":
        print("applying frame effect...")
        effect_mask = sieve.File(path="assets/square.jpg")

        return apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15)


    raise ValueError(f"Effect {effect} not supported")


if __name__ == "__main__":
    video_path = "duckling.mp4"
    subject = "duckling"
    effect = "frame"

    video = sieve.File(path=video_path)

    output = add_effect(video, subject, effect)

    shutil.move(output.path, os.path.join("outputs", f"{effect}_{video_path}"))

