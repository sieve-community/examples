import sieve
import cv2
import shutil
import os
import zipfile
import tempfile


def is_video(file: sieve.File):
    file_path = file.path

    video_formats = ['mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv']

    if file_path.split(".")[-1] in video_formats:
        return True

    return False


def get_first_frame(video: sieve.File):
    video_path = video.path

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('first_frame.png', frame)
    else:
        raise Exception("Failed to read the video; empty or does not exist")

    frame = sieve.File(path='first_frame.png')
    cap.release()

    return frame


def get_object_bbox(image: sieve.File, object_name: str):
    yolo = sieve.function.get('sieve/yolov8')

    response = yolo.run(
        file=image,
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]
    bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]

    return bounding_box

metadata = sieve.Metadata(
    title="text-to-segment",
    description="Text prompt SAM2 to segment a video or image.",
    readme=open("README.md").read(),
    image=sieve.File(path="duck_silhouette.png")
)


@sieve.function(
    name="text-to-segment",
    python_packages=["opencv-python"],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    metadata=metadata
)
def segment(file: sieve.File, object_name: str):
    """
    :param file: photo or video to segment
    :param object_name: the object you wish to segment
    """
    sam = sieve.function.get("sieve/sam2")

    if is_video(file):
        image = get_first_frame(file)
    else:
        image = file

    print("fetching bounding box...")
    box = get_object_bbox(image, object_name)

    sam_prompt = {
        "object_id": 1,   # id to track the object
        "frame_index": 0, # first frame (if it's a video)
        "box": box        # bounding box [x1, y1, x2, y2]
    }

    sam_out = sam.run(
        file=file,
        prompts=[sam_prompt],
        model_type="tiny",
        debug_masks=False
    )

    return sam_out


def zip_to_mp4(frames_zip: sieve.File):
    """
    convert zip file of frames to an mp4
    """
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



if __name__ == "__main__":

    video_path = "duckling.mp4"
    text_prompt = "duckling"

    video = sieve.File(path=video_path)
    sam_out = segment(video, text_prompt)

    mask = zip_to_mp4(sam_out['masks'])

    os.makedirs("outputs", exist_ok=True)
    shutil.move(mask.path, os.path.join("outputs", f"segment_{video_path}"))
