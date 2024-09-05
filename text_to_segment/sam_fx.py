import sieve
import os
import subprocess
import shutil
import cv2
import numpy as np

from typing import Literal

from blending import blend_to_background
from utils import splice_audio
from text_to_segment import segment
from utils import get_first_frame, resize_and_crop, resize_with_padding

from config import CACHE

def reencode_video(video: sieve.File):
    video_path = video.path
    # cmd = f"ffmpeg -i {video_path}  -y -nostdin -c:v libx264 -preset fast -pix_fmt yuv420p -crf 23 reencoded.mp4"
    # os.system(cmd)
    cmd = ["ffmpeg", "-i", video_path, "-loglevel", "error", "-y", "-nostdin", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-crf", "23", "reencoded.mp4"]

    subprocess.run(cmd, check=True)

    shutil.move("reencoded.mp4", video_path)

    return sieve.File(path=video_path)


def apply_shape_effect(video: sieve.File, mask_video: sieve.File, effect_mask: sieve.File, mask_scale=1.):

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


def gaussian_blur(image: np.array):
    return cv2.GaussianBlur(image, (15, 15), 0)


def apply_color_filter(image: np.array, color: tuple, intensity: float = 0.5):
    # Ensure the color is in BGR format for OpenCV
    b, g, r = color

    # Create a color overlay
    overlay = np.full(image.shape, (b, g, r), dtype=np.uint8)

    # Blend the original image with the color overlay
    filtered = cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)

    return filtered



# def dim_brightness(image: np.array):

#     dimmed = image.astype(np.float32) * 0.5

#     return dimmed.astype(np.uint8)


# def apply_filter(video: sieve.File, mask_video: sieve.File, filter_fn: callable, to_foreground=True):
#     video_reader = cv2.VideoCapture(video.path)
#     mask_reader = cv2.VideoCapture(mask_video.path)

#     frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = video_reader.get(cv2.CAP_PROP_FPS)

#     output_path = "output.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     while True:
#         ret_video, frame_video = video_reader.read()
#         ret_mask, frame_mask = mask_reader.read()

#         if not ret_video or not ret_mask:
#             break

#         if not len(frame_mask.shape) == 3:
#             frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
#             frame_mask = np.expand_dims(frame_mask, axis=2)
#             frame_mask = np.repeat(frame_mask, 3, axis=2)

#         frame_mask = frame_mask > 0
#         if not to_foreground:
#             frame_mask = ~frame_mask

#         try:
#             frame_filtered = filter_fn(frame_video)
#             frame_video[frame_mask] = frame_filtered[frame_mask]
#         except:
#             breakpoint()

#         output_writer.write(frame_video)

#     video_reader.release()
#     mask_reader.release()
#     output_writer.release()

#     return sieve.File(path=output_path)

def dim_brightness(image: np.array):
    dimmed = image.astype(np.float32) * 0.5
    return dimmed.astype(np.uint8)

def apply_filter(video: sieve.File, mask_video: sieve.File, filter_fn: callable, to_foreground=True):
    video_reader = cv2.VideoCapture(video.path)
    mask_reader = cv2.VideoCapture(mask_video.path)

    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret_video, frame_video = video_reader.read()
        ret_mask, frame_mask = mask_reader.read()

        if not ret_video or not ret_mask:
            break

        if len(frame_mask.shape) == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the mask
        frame_mask = cv2.GaussianBlur(frame_mask, (15, 15), 0)
        
        # Normalize mask to range 0-1
        frame_mask = frame_mask.astype(float) / 255.0
        
        if not to_foreground:
            frame_mask = 1 - frame_mask

        # Apply filter to the entire frame
        frame_filtered = filter_fn(frame_video)
        
        # Blend the filtered and original frames using the mask
        frame_mask = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2)
        blended = frame_filtered * frame_mask + frame_video * (1 - frame_mask)

        output_writer.write(blended.astype(np.uint8))


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
    effect: Literal[
        "circle",
        "spotlight",
        "frame",
        "retro solar",
        "focus",
        "blur",
        "red",
        "green",
        "blue",
        "yellow",
        "orange"]
):
    """
    :param video: input video
    :param subject: subject of video
    :param effect: effect to overlay
    """

    if CACHE and os.path.exists("mask.mp4"):
        mask_video = sieve.File(path="mask.mp4")
    else:
        print("segmenting...")
        mask_video = segment(video, subject, return_mp4=True)

        if CACHE:
            shutil.copy(mask_video.path, "mask.mp4")


    if effect == "retro solar":
        print("applying retro solar effect...")
        effect_mask = sieve.File(path="assets/rays.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask)

    elif effect == "circle":
        print("applying circle effect...")
        effect_mask = sieve.File(path="assets/circle.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.2)

    elif effect == "spotlight":
        print("applying splotlight effect...")
        effect_mask = sieve.File(path="assets/spot.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15)

    elif effect == "frame":
        print("applying frame effect...")
        effect_mask = sieve.File(path="assets/square.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15)


    elif effect == "focus":
        print("applying focus effect...")
        out = apply_filter(video, mask_video, dim_brightness, to_foreground=False)

    elif effect == "blur":
        print("applying blur effect...")
        out = apply_filter(video, mask_video, gaussian_blur, to_foreground=False)

    elif effect == "red":
        print("applying red effect...")
        red_filter = lambda img: apply_color_filter(img, (0, 0, 255), 0.3)
        out = apply_filter(video, mask_video, red_filter, to_foreground=True)

    elif effect == "green":
        print("applying green effect...")
        green_filter = lambda img: apply_color_filter(img, (113, 179, 60), 0.3)
        out = apply_filter(video, mask_video, green_filter, to_foreground=True)

    elif effect == "blue":
        print("applying blue effect...")
        blue_filter = lambda img: apply_color_filter(img, (255, 0, 0), 0.3)
        out = apply_filter(video, mask_video, blue_filter, to_foreground=True)

    elif effect == "yellow":
        print("applying yellow effect...")
        yellow_filter = lambda img: apply_color_filter(img, (0, 255, 255), 0.3)
        out = apply_filter(video, mask_video, yellow_filter, to_foreground=True)

    elif effect == "orange":
        print("applying orange effect...")
        orange_filter = lambda img: apply_color_filter(img, (0, 165, 255), 0.3)
        out = apply_filter(video, mask_video, orange_filter, to_foreground=True)

    else:
        raise ValueError(f"Effect {effect} not supported")

    return reencode_video(out)


def run_all(video_path, subject):

    video = sieve.File(path=video_path)

    for effect in ["circle", "spotlight", "frame", "retro solar", "focus", "blur", "red", "green", "blue", "yellow", "orange"]:
        output = add_effect(video, subject, effect)

        shutil.move(output.path, os.path.join("outputs", f"{effect}_{video_path}"))


if __name__ == "__main__":
    video_path = "duckling.mp4"
    subject = "duckling"
    effect = "blur"

    video = sieve.File(path=video_path)

    # output = add_effect(video, subject, effect)
    # shutil.move(output.path, os.path.join("outputs", f"{effect}_{video_path}"))

    run_all(video_path, subject)
