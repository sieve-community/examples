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

import config
from concurrent.futures import ThreadPoolExecutor, as_completed


def reencode_video(video: sieve.File):
    video_path = video.path
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


def apply_color_filter(image: np.array, color: tuple, intensity: float = 0.5):
    # Ensure the color is in BGR format for OpenCV
    b, g, r = color

    # Create a color overlay
    overlay = np.full(image.shape, (b, g, r), dtype=np.uint8)

    # Blend the original image with the color overlay
    filtered = cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)

    return filtered


def dim_brightness(image: np.array, brightness=0.5):
    dimmed = image.astype(np.float32) * brightness
    return dimmed.astype(np.uint8)


def process_frame(frame_video, frame_mask, filter_fn, to_foreground):
    if len(frame_mask.shape) == 3:
        frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

    frame_mask = cv2.GaussianBlur(frame_mask, (15, 15), 0)
    frame_mask = frame_mask.astype(float) / 255.0

    if not to_foreground:
        frame_mask = 1 - frame_mask

    frame_filtered = filter_fn(frame_video)

    frame_mask = np.repeat(frame_mask[:, :, np.newaxis], 3, axis=2)
    blended = frame_filtered * frame_mask + frame_video * (1 - frame_mask)

    return blended.astype(np.uint8)

def apply_filter(video: sieve.File, mask_video: sieve.File, filter_fn: callable, to_foreground=True):
    video_reader = cv2.VideoCapture(video.path)
    mask_reader = cv2.VideoCapture(mask_video.path)

    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def read_frames():
        while True:
            ret_video, frame_video = video_reader.read()
            ret_mask, frame_mask = mask_reader.read()
            if not ret_video or not ret_mask:
                break
            yield frame_video, frame_mask

    with ThreadPoolExecutor() as executor:
        futures = []
        for frame_video, frame_mask in read_frames():
            future = executor.submit(process_frame, frame_video, frame_mask, filter_fn, to_foreground)
            futures.append(future)

        for future in futures:
            blended_frame = future.result()
            output_writer.write(blended_frame)

    video_reader.release()
    mask_reader.release()
    output_writer.release()

    return sieve.File(path=output_path)

def get_mask(video: sieve.File, subject: str):

    if config.CACHE and os.path.exists("mask.mp4"):
        mask_video = sieve.File(path="mask.mp4")
    else:
        print("segmenting...")
        mask_video = segment(video, subject, return_mp4=True)

        if config.CACHE:
            shutil.copy(mask_video.path, "mask.mp4")

    return mask_video


# FOCUS EFFECT ################################################################################

focus_readme = """
# Focus Effect

## Description

Dim the background to highlight the subject

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to apply the effect to
- `brightness` (Literal["0.25", "0.5", "0.75"], optional): The brightness of the background. Default is `"0.25"`.

## Examples

```python
focus = sieve.function.get("sieve-internal/sam2-focus")

video = File(path="duckling.mp4")
subject = "duckling"

output = focus.run(video, subject, brightness="0.5")
```

## How does it work?
- The video is segmented using a separate function to create a mask of the subject.
- A simple dimming function is applied to each frame, where the brightness level is specified by the user (0.25, 0.5, or 0.75).
- The apply_filter function then uses this dimming function and the mask to blend the dimmed and original frames.
- This creates a video where the subject remains bright while the background is dimmed to the specified level.

"""


metadata = sieve.Metadata(
    name="Focus",
    description="Dim the background to highlight the subject",
    image=sieve.File(path=os.path.join("thumbnails", "focus_0-5_duckling.png")),
    readme=focus_readme
)

@sieve.function(
    name="sam2-focus",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def focus(
        video: sieve.File,
        subject: str,
        brightness: Literal["0.25", "0.5", "0.75"] = "0.25"
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to apply the effect to
    :param brightness: The brightness of the background
    """


    mask_video = get_mask(video, subject)
    dim = lambda img: dim_brightness(img, float(brightness))
    out = apply_filter(video, mask_video, dim, to_foreground=False)

    return reencode_video(out)

# CALLOUT EFFECTS ################################################################################


callout_readme = """
# Callout Effects

## Description

Places a shape behind the subject to make it pop out!

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to apply the effect to
- `effect` (Literal["retro solar", "circle", "spotlight", "frame"], optional): The effect to apply. Default is `"circle"`.
- `effect_scale` (float, optional): Adjust the size of the shape, e.g. 2.0 will double the size of the effect. Default is `1.0`.

## Examples

```python
callout = sieve.function.get("sieve-internal/sam2-callout")

video = File(path="duckling.mp4")
subject = "duckling"

output = callout.run(video, subject, effect="retro solar")
```

## Effects

- `"retro solar"`: A retro solar effect applied behind the subject
- `"circle"`: A circular effect applied behind the subject
- `"spotlight"`: A spotlight effect applied behind the subject
- `"frame"`: A frame effect applied behind the subject

## How does it work?
- A pre-made image (e.g., circle, spotlight) is used as the effect mask.
- For each frame, the center of the subject is calculated applying OpenCV's moments function to the segmentation mask.
- The center position is smoothed using a simple exponential moving average to reduce jitter in the mask position
- The effect mask is then translated to this center position using a translation matrix and OpenCV's warpAffine function.
- The translated mask is combined with the subject mask and used to blend the effect into the original frame.


"""

metadata = sieve.Metadata(
    name="Callout",
    description="Highlight a subject in a video with a callout effect",
    image=sieve.File(path=os.path.join("thumbnails", "retro_solar_duckling.png")),
    readme=callout_readme
)

@sieve.function(
    name="sam2-callout",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def callout(
        video: sieve.File, 
        subject: str,
        effect: Literal["retro solar", "circle", "spotlight", "frame"] = "retro solar",
        effect_scale: float = 1.0
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to apply the effect to
    :param effect: The effect to apply
    :param effect_scale: The scale of the effect
    """

    mask_video = get_mask(video, subject)

    if effect == "retro solar":
        effect_mask = sieve.File(path="assets/rays.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, effect_scale)

    elif effect == "circle":
        effect_mask = sieve.File(path="assets/circle.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.2*effect_scale)

    elif effect == "spotlight":
        effect_mask = sieve.File(path="assets/spot.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15*effect_scale)

    elif effect == "frame":
        effect_mask = sieve.File(path="assets/square.jpg")

        out = apply_shape_effect(video, mask_video, effect_mask, mask_scale=0.15*effect_scale)

    else:
        raise ValueError(f"Effect {effect} not supported")

    return reencode_video(out)


# COLOR FILTERS ################################################################################

color_filter_readme = """
# Color Filter

## Description

Highlight an object in the video with a colour filter!

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to apply the effect to
- `color` (Literal["red", "green", "blue", "yellow", "orange"]): The color to apply the effect with
- `intensity` (float, optional): The intensity of the effect. Default is `0.5`.

## Examples

```python
color_filter = sieve.function.get("sieve-internal/sam2-color-filter")

video = File(path="duckling.mp4")
subject = "duckling"

output = color_filter.run(video, subject, color="red")
```

## Colors

- `"red"`: Apply a red color filter
- `"green"`: Apply a green color filter
- `"blue"`: Apply a blue color filter
- `"yellow"`: Apply a yellow color filter
- `"orange"`: Apply an orange color filter

## How does it work?
- A simple color overlay function is defined that creates a full-frame overlay of the specified color and blends it with the original image using OpenCV's addWeighted function.
- This function is applied to each frame, with the color and intensity specified by the user.
- The apply_filter function then uses the segmentation mask to apply this colored version only to the subject area of each frame.

"""



metadata = sieve.Metadata(
    name="Color Filter",
    description="Apply a color filter to a video",
    image=sieve.File(path=os.path.join("thumbnails", "red_duckling.png")),
    readme=color_filter_readme
)

@sieve.function(
    name="sam2-color-filter",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def color_filter(
        video: sieve.File,
        subject: str,
        color: Literal["red", "green", "blue", "yellow", "orange"] = "red",
        intensity: float = 0.5
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to apply the effect to
    :param color: The color to apply the effect with
    :param intensity: The intensity of the effect
    """

    mask_video = get_mask(video, subject)

    if color == "red":
        print("applying red effect...")
        color_filter_fn = lambda img: apply_color_filter(img, (0, 0, 255), intensity)

    elif color == "green":
        print("applying green effect...")
        color_filter_fn = lambda img: apply_color_filter(img, (113, 179, 60), intensity)

    elif color == "blue":
        print("applying blue effect...")
        color_filter_fn = lambda img: apply_color_filter(img, (255, 0, 0), intensity)

    elif color == "yellow":
        print("applying yellow effect...")
        color_filter_fn = lambda img: apply_color_filter(img, (0, 255, 255), intensity)

    elif color == "orange":
        print("applying orange effect...")
        color_filter_fn = lambda img: apply_color_filter(img, (0, 165, 255), intensity)

    else:
        raise ValueError(f"Color {color} not supported")

    out = apply_filter(video, mask_video, color_filter_fn, to_foreground=True)

    return reencode_video(out)


# BLUR EFFECT ################################################################################


blur_readme = """
# Blur Effect

## Description

Blur the background of a video

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to apply the effect to
- `blur_amount` (Literal["low", "medium", "high"]): The amount of blur to apply

## Examples

```python
blur = sieve.function.get("sieve-internal/sam2-blur")

video = File(path="duckling.mp4")
subject = "duckling"

output = blur.run(video, subject, blur_amount="high")
```

## Blur Amounts

- `"low"`: Apply a low amount of blur
- `"medium"`: Apply a medium amount of blur
- `"high"`: Apply a high amount of blur

## How does it work?
- A Gaussian blur function is defined based on the blur amount, using OpenCV's GaussianBlur function with different kernel sizes (15 for low, 25 for medium, or 35 for high blur).
- This blur is applied to the entire frame.
- The apply_filter function then uses the inverted segmentation mask to blend the blurred background with the original subject.

"""


metadata = sieve.Metadata(
    name="Blur",
    description="Blur the background of a video",
    image=sieve.File(path=os.path.join("thumbnails", "high_blur_duckling.png")),
    readme=blur_readme
)

@sieve.function(
    name="sam2-blur",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def blur(
        video: sieve.File,
        subject: str,
        blur_amount: Literal["low", "medium", "high"] = "medium"
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to apply the effect to
    :param blur_amount: The amount of blur to apply
    """

    mask_video = get_mask(video, subject)

    if blur_amount == "low":
        print("applying low blur effect...")
        blur_filter = lambda img: cv2.GaussianBlur(img, (15, 15), 0)
    elif blur_amount == "medium":
        print("applying medium blur effect...")
        blur_filter = lambda img: cv2.GaussianBlur(img, (25, 25), 0)
    elif blur_amount == "high":
        print("applying high blur effect...")
        blur_filter = lambda img: cv2.GaussianBlur(img, (35, 35), 0)

    out = apply_filter(video, mask_video, blur_filter, to_foreground=False)


    return reencode_video(out)


# SELECTIVE COLOR EFFECT ################################################################################


selective_color_readme = """
# Selective Color Effect

## Description

Keep the subject in color while turning the background black and white.

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to keep in color

## Examples

```python
selective_color = sieve.function.get("sieve-internal/sam2-selective-color")

video = File(path="duckling.mp4")
subject = "duckling"

output = selective_color.run(video, subject)
```

## How does it work?
- The video is segmented using a separate function to create a mask of the subject.
- For each frame, two versions are created: a full-color version (the original) and a grayscale version.
- The segmentation mask is used to blend these two versions: the color version for the subject and the grayscale version for the background.
- This creates a striking contrast between the colorful subject and the monochrome background.
"""

selective_color_metadata = sieve.Metadata(
    name="Selective Color",
    description="Keep the subject in color while turning the background black and white",
    image=sieve.File(path=os.path.join("thumbnails", "selective_color_duckling.png")),
    readme=selective_color_readme
)

@sieve.function(
    name="sam2-selective-color",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=selective_color_metadata
)
def selective_color(
        video: sieve.File,
        subject: str
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to keep in color
    """
    mask_video = get_mask(video, subject)

    def color_filter(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    out = apply_filter(video, mask_video, color_filter, to_foreground=False)
    return reencode_video(out)


# PIXELATE EFFECT ################################################################################

pixelate_readme = """
# Pixelate Effect

## Description

Apply a pixelated look to the subject while keeping the background clear.

## Parameters

- `video` (File): The video file to apply the effect to
- `subject` (str): The subject to pixelate
- `pixel_size` (int, optional): The size of the pixels in the pixelation effect. Default is 20.

## Examples

```python
pixelate = sieve.function.get("sieve-internal/sam2-pixelate")

video = File(path="duckling.mp4")
subject = "duckling"

output = pixelate.run(video, subject, pixel_size=15)
```

## How does it work?
- The video is segmented using a separate function to create a mask of the subject.
- A pixelation function is applied to each frame, which reduces the resolution and then enlarges it back, creating a blocky look.
- The segmentation mask is used to apply this pixelated version only to the subject area of each frame.
- This creates an interesting visual where the subject appears blocky and pixelated while the background remains clear and detailed.
"""

pixelate_metadata = sieve.Metadata(
    name="Pixelate",
    description="Apply a pixelated look to the subject while keeping the background clear",
    image=sieve.File(path=os.path.join("thumbnails", "pixelate_25_duckling.png")),
    readme=pixelate_readme
)

@sieve.function(
    name="sam2-pixelate",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=pixelate_metadata
)
def pixelate(
        video: sieve.File,
        subject: str,
        pixel_size: Literal["15", "20", "25"] = "20"
):
    """
    :param video: The video file to apply the effect to
    :param subject: The subject to pixelate
    :param pixel_size: The size of the pixels in the pixelation effect
    """
    mask_video = get_mask(video, subject)

    pixel_size = int(pixel_size)

    def pixelate_filter(frame):
        height, width = frame.shape[:2]
        temp = cv2.resize(frame, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    out = apply_filter(video, mask_video, pixelate_filter, to_foreground=True)
    return reencode_video(out)


def run_all(video_path, subject):
    config.CACHE = True

    video = sieve.File(path=video_path)

    os.makedirs("outputs", exist_ok=True)

    # Focus
    for brightness in ["0.25", "0.5", "0.75"]:
        out_path = os.path.join("outputs", f"focus_{brightness.replace('.', '-')}_{video_path}")
        if os.path.exists(out_path):
            continue

        output = focus(video, subject, brightness)
        shutil.move(output.path, out_path)

    # Callout
    for effect in ["circle", "spotlight", "frame", "retro solar"]:
        out_path = os.path.join("outputs", f"{effect.replace(' ', '_')}_{video_path}")
        if os.path.exists(out_path):
            continue

        output = callout(video, subject, effect)
        shutil.move(output.path, out_path)

    # Color Filter
    for color in ["red", "green", "blue", "yellow", "orange"]:
        out_path = os.path.join("outputs", f"{color}_{video_path}")
        if os.path.exists(out_path):
            continue

        output = color_filter(video, subject, color)
        shutil.move(output.path, out_path)

    # Blur
    for blur_amount in ["low", "medium", "high"]:
        out_path = os.path.join("outputs", f"{blur_amount}_blur_{video_path}")
        if os.path.exists(out_path):
            continue

        output = blur(video, subject, blur_amount)
        shutil.move(output.path, out_path)

    # Selective Color
    out_path = os.path.join("outputs", f"selective_color_{video_path}")
    if not os.path.exists(out_path):
        output = selective_color(video, subject)
        shutil.move(output.path, out_path)

    # Pixelate
    for pixel_size in [15, 20, 25, 50]:
        out_path = os.path.join("outputs", f"pixelate_{pixel_size}_{video_path}")
        if os.path.exists(out_path):
            continue

        output = pixelate(video, subject, pixel_size)
        shutil.move(output.path, out_path)


if __name__ == "__main__":
    video_path = "duckling.mp4"
    subject = "duckling"

    video = sieve.File(path=video_path)

    run_all(video_path, subject)
