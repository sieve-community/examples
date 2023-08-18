from sieve_model import VideoRetalker, LipsyncInputs
from sieve import Video, Audio

import os

os.environ["stabilize_expression"] = "true"
os.environ["reference_enhance"] = "false"
os.environ["gfpgan_enhance"] = "true"
os.environ["post_enhance"] = "true"

retalker = VideoRetalker()

retalker.__predict__(
    LipsyncInputs(
        video=Video(
            path="/home/ubuntu/experiments/assets/lucy.mp4"
        ),
        audio=Audio(
            path="/home/ubuntu/experiments/assets/01.wav"
        )
    )
)