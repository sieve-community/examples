import sieve
import cv2
import shutil
import os
import zipfile
import tempfile
import numpy as np



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
    cap.release()

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


def zip_to_mp4(frames_zip):
    output_path = "output_video.mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(frames_zip.path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split('_')[1]))

        first_frame = cv2.imread(os.path.join(temp_dir, images[0]))
        height, width, layers = first_frame.shape
        frame_size = (width, height)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

        # Loop through the images and write them to the video
        for image in images:
            img_path = os.path.join(temp_dir, image)
            frame = cv2.imread(img_path)
            out.write(frame)

    out.release()
    return sieve.File(path=output_path)




def resize_and_crop(image, target_width, target_height):
    image_height, image_width = image.shape[:2]

    target_aspect = target_width / target_height
    image_aspect = image_width / image_height

    if image_aspect > target_aspect:
        new_height = target_height
        new_width = int(image_aspect * new_height)
    else:
        new_width = target_width
        new_height = int(new_width / image_aspect)

    resized_image = cv2.resize(image, (new_width, new_height))

    crop_x = (new_width - target_width) // 2
    crop_y = (new_height - target_height) // 2

    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image


def blend_to_background(object_video, mask_video, background_img):
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


def splice_audio(video, audio):
    spliced_path = "spliced.mp4"
    cmd = f"ffmpeg -y -nostdin -loglevel error -i {video.path} -i {audio.path} -c:v copy -c:a aac {spliced_path}"
    os.system(cmd)

    return sieve.File(path=spliced_path)


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




