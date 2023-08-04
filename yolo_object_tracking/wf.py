import sieve
from typing import Dict, Tuple
from yolo import Yolo
from tracker import SORT
from visualizer import draw_boxes

@sieve.workflow(name="object_tracking")
def yolosplit(video: sieve.Video) -> Dict:
    '''
    :param video: Video of objects to be tracked
    :return: Dictionary of tracked objects
    '''
    images = sieve.reference("sieve/video-splitter")(video)
    yolo_outputs = Yolo()(images)
    return SORT(yolo_outputs)

@sieve.workflow(name="object_tracking_visualize")
def yolo_visualize(video: sieve.Video) -> Tuple[sieve.Video, Dict]:
    '''
    :param video: Video of objects to be tracked
    :return: Original video with tracked objects overlayed and a dictionary of tracked objects
    '''

    images = sieve.reference("sieve/video-splitter")(video)
    yolo_outputs = Yolo()(images)
    visualized = draw_boxes(images, yolo_outputs)
    combined = sieve.reference("sieve/frame-combiner")(visualized)
    return combined, SORT(yolo_outputs)
