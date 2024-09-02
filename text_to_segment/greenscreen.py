import sieve
import os
import shutil

from blending import blend_to_background
from utils import splice_audio
from text_to_segment import segment


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
    """
    :param video: input video to segment
    :param background: image to use as background
    :param subjects: comma-separated list of objects to track in the video
    """

    print("segmenting...")
    mask_video = segment(video, subject, return_mp4=True)

    print("blending...")
    blended_vid = blend_to_background(video, mask_video, background)

    out = splice_audio(blended_vid, video)

    return out


if __name__ == "__main__":

    video_path = "duckling.mp4"
    subject = "duckling"
    bg_path = "galaxy.jpg"

    video = sieve.File(path=video_path)
    background = sieve.File(path=bg_path)

    os.makedirs("outputs", exist_ok=True)

    out = background_replace(video, background, subject)

    shutil.move(out.path, os.path.join("outputs", f"{bg_path.split('/')[-1].split('.')[0]}_{video_path}"))

