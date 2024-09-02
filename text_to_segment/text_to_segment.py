import sieve
import shutil
import os

from utils import is_video, get_first_frame, zip_to_mp4
from bbox import get_object_bbox



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
def segment(file: sieve.File, object_name: str, return_mp4: bool = False):
    """
    :param file: photo or video to segment
    :param object_name: the object you wish to segment
    :param return_mp4: if True, return only an MP4 video of the segmentation masks
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

    if return_mp4:
        return zip_to_mp4(sam_out["masks"])

    return sam_out




if __name__ == "__main__":

    video_path = "duckling.mp4"
    text_prompt = "duckling"

    video = sieve.File(path=video_path)
    sam_out = segment(video, text_prompt)

    mask = zip_to_mp4(sam_out['masks'])

    os.makedirs("outputs", exist_ok=True)
    shutil.move(mask.path, os.path.join("outputs", f"segment_{video_path}"))
