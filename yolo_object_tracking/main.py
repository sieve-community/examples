import sieve
from typing import Dict, Tuple
from yolo import Yolo
from tracker import SORT
from visualizer import draw_boxes

metadata = sieve.Metadata(
    title="Track Objects in Video",
    description="Run hyperparallelized object tracking on video.",
    code_url="https://github.com/sieve-community/examples/tree/main/yolo_object_tracking/main.py",
    tags=["Tracking", "Video", "Detection"],
    readme=open("README.md", "r").read(),
)


@sieve.workflow(name="object_tracking", metadata=metadata)
def yolosplit(video: sieve.Video) -> Dict:
    """
    :param video: Video of objects to be tracked
    :return: Dictionary of tracked objects
    """
    images = sieve.reference("sieve/video-splitter")(video)
    yolo_outputs = Yolo()(images)
    return SORT(yolo_outputs)


@sieve.workflow(name="object_tracking_visualize", metadata=metadata)
def yolo_visualize(video: sieve.Video) -> Tuple[sieve.Video, Dict]:
    """
    :param video: Video of objects to be tracked
    :return: Original video with tracked objects overlayed and a dictionary of tracked objects
    """

    images = sieve.reference("sieve/video-splitter")(video)
    yolo_outputs = Yolo()(images)
    visualized = draw_boxes(images, yolo_outputs)
    combined = sieve.reference("sieve/frame-combiner")(visualized)
    return combined, SORT(yolo_outputs)


if __name__ == "__main__":
    sieve.push(
        yolosplit,
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/bike.mp4"
            }
        },
    )
