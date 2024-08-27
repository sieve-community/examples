import sieve
import cv2
import numpy as np

from utils import resize_and_crop


def blend_to_background(object_video, mask_video, background_img):
    """
    superimpose `object_video` onto `background_img` using `mask_video`

    assumes that `mask_video` frames correspond 1-1 with `object_video` frames
    (but framerate doesn't matter)
    """
    object_video = cv2.VideoCapture(object_video.path)
    mask_video = cv2.VideoCapture(mask_video.path)
    background = cv2.imread(background_img.path)

    output_path = "blended_output.mp4"

    frame_width = int(object_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(object_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = object_video.get(cv2.CAP_PROP_FPS)

    background = resize_and_crop(background, frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret_obj, obj_frame = object_video.read()
        ret_mask, mask_frame = mask_video.read()

        if not ret_obj or not ret_mask:
            break

        if len(mask_frame.shape) == 3:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        mask = mask_frame.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)

        blended_frame = (obj_frame * mask + background * (1 - mask)).astype(np.uint8)

        output_video.write(blended_frame)

    object_video.release()
    mask_video.release()
    output_video.release()

    return sieve.File(path=output_path)
