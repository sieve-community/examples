import sieve
from typing import Dict
from yolo import Yolo
from tracker import SORT
from splitter import VideoSplitter

@sieve.workflow(name="yolo_object_tracking")
def yolosplit(video: sieve.Video) -> Dict:
    images = VideoSplitter(video)
    yolo_outputs = Yolo()(images)
    return SORT(yolo_outputs)
