import sieve
from typing import Dict
from bg import U2NetMask, U2NetBlur
from combiner import frame_combine
from splitter import VideoSplitter

@sieve.workflow(name="video_background_mask")
def background_mask(video: sieve.Video) -> Dict:
    images = VideoSplitter(video)
    masks = U2NetMask()(images)
    return frame_combine(masks)

@sieve.workflow(name="video_background_blur")
def background_blur(video: sieve.Video) -> Dict:
    images = VideoSplitter(video)
    masks = U2NetBlur()(images)
    return frame_combine(masks)
