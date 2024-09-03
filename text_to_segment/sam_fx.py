import sieve
import os
import shutil

import cv2

from blending import blend_to_background
from utils import splice_audio
from text_to_segment import segment


import cv2
import numpy as np


def resize_to(mask, target_shape):
    """ Resizes the mask to the target image dimensions. """

    # get larger target dimension
    target_dim = max(target_shape)

    return cv2.resize(mask, (target_dim, target_dim))  # Width and height are reversed in cv2.resize


def translate_mask(mask, target_center, original_shape):

    """ Translates the mask's center of mass to the target center coordinate. """
    # Calculate current center of mass of the mask
    M = cv2.moments(mask, binaryImage=True)
    current_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

    # Translation vector from current center to target center
    translation = (target_center[1] - current_center[1], target_center[0] - current_center[0])

    # Translation matrix
    translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])

    # Translate and handle boundaries by padding if necessary
    padded_mask = cv2.copyMakeBorder(mask, max(-translation[1], 0), max(translation[1], 0),
                                     max(-translation[0], 0), max(translation[0], 0),
                                     cv2.BORDER_CONSTANT, value=0)
    
    # Apply the translation
    translated_mask = cv2.warpAffine(padded_mask, translation_matrix, (padded_mask.shape[1], padded_mask.shape[0]))

    # Crop back to the original image size
    start_y = max(translation[1], 0)
    start_x = max(translation[0], 0)
    translated_mask = translated_mask[start_y:start_y + original_shape[0], start_x:start_x + original_shape[1]]

    return translated_mask



def center_mask_on_coordinate(mask, target_center, target_shape):
    """ Centers the mask on the given coordinate in the context of a target shape. """
    # Resize mask to match the target shape
    resized_mask = resize_to(mask, target_shape)
    
    # Translate resized mask to center it on the target center
    centered_mask = translate_mask(resized_mask, target_center, target_shape)

    return centered_mask




@sieve.function(
    name="video-effect",
    python_packages=["opencv-python"],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ]
)
def add_effect(
    video: sieve.File,
    effect: sieve.File,
    subject: str,
):
    """
    :param video: input video to segment
    :param background: image to use as background
    :param subjects: comma-separated list of objects to track in the video
    """

    print("segmenting...")
    mask_video = segment(video, subject, return_mp4=True)


    return


if __name__ == "__main__":

    # video_path = "duckling.mp4"
    # subject = "duckling"
    # effect_path = "rays.jpg"

    # video = sieve.File(path=video_path)
    # effect = sieve.File(path=effect_path)

    # os.makedirs("outputs", exist_ok=True)

    # out = add_effect(video, effect, subject)

    # shutil.move(out.path, os.path.join("outputs", f"{effect_path.split('/')[-1].split('.')[0]}_{video_path}"))

    from utils import get_first_frame
    from greenscreen import background_replace

    video_path = "duckling.mp4"

    video = sieve.File(path=video_path)
    first_frame = get_first_frame(video)

    effect_path = "rays.jpeg"

    cap = cv2.VideoCapture(video.path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    from utils import resize_and_crop

    effect = sieve.File(path=effect_path)
    effect_mask = cv2.imread(effect.path)
    if len(effect_mask.shape) == 3:
        effect_mask = cv2.cvtColor(effect_mask, cv2.COLOR_BGR2GRAY)

    size = 2*max(frame_width, frame_height)
    effect_mask = resize_and_crop(effect_mask, size, size)

    # effect_mask = resize_and_crop(effect_mask, frame_width, frame_height)
    _, effect_mask = cv2.threshold(effect_mask, 127, 255, cv2.THRESH_BINARY)

    output_path = "effect.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    mask_vid_path = "outputs/segment_duckling.mp4"
    assert os.path.exists(mask_vid_path)
    mask_cap = cv2.VideoCapture(mask_vid_path)

    prev_current_center = None

    while True:
        ret_obj, obj_frame = cap.read()
        ret_mask, mask_frame = mask_cap.read()


        if not ret_obj:
            break

        if len(mask_frame.shape) == 3:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)


        M = cv2.moments(mask_frame, binaryImage=True)

        current_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if prev_current_center is None:
            prev_current_center = current_center

        beta = 0.2
        current_center = (int((1.-beta)*prev_current_center[0] + beta*current_center[0]), int(0.8*prev_current_center[1] + 0.2*current_center[1]))
        prev_current_center = current_center

        translation = (current_center[0] - size//2, current_center[1] - size//2)

        translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
        translated_mask = cv2.warpAffine(effect_mask, translation_matrix, (mask_frame.shape[1], mask_frame.shape[0]))

        # translated_mask = resize_and_crop(translated_mask, frame_width, frame_height)

        # combined_mask = cv2.bitwise_not(mask_frame)
        # combined_mask = mask_frame
        combined_mask = cv2.bitwise_or(mask_frame, translated_mask)

        mask = combined_mask

        cv2.imwrite("mask.jpg", mask)

        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)

        new_frame = obj_frame * mask + (1 - mask) * 255.

        output_video.write(new_frame.astype(np.uint8))


    cap.release()
    mask_cap.release()
    output_video.release()

    # cv2.imwrite("centered_mask.jpg", centered_mask)
    
