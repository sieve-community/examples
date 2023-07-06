import sieve
from typing import Dict, Tuple
from yolo import Yolo
from tracker import SORT
from splitter import VideoSplitter
from visualizer import draw_boxes

@sieve.workflow(name="yolo_object_tracking")
def yolosplit(video: sieve.Video) -> Dict:
    images = VideoSplitter(video)
    yolo_outputs = Yolo()(images)
    return SORT(yolo_outputs)

@sieve.workflow(name="yolo_object_tracking_visualize")
def yolo_visualize(video: sieve.Video) -> Tuple[sieve.Video, Dict]:
    images = VideoSplitter(video)
    yolo_outputs = Yolo()(images)
    visualized = draw_boxes(images, yolo_outputs)
    combined = sieve.reference("sieve-developer/frame-combiner")(visualized)
    sink = sieve.reference("sieve-developer/general-sink")(combined)
    return sink, SORT(yolo_outputs)


if __name__ == "__main__":
    sieve.push(
        yolosplit,
        {
            "video": sieve.Video(
                url="https://storage.googleapis.com/sieve-test-videos-central/01-lebron-dwade.mp4"
            )
        },
        wait=True,
    )
