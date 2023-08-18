import sieve
from typing import Dict, List

readme_content = open("README.md", "r").read()
wf_metadata = sieve.Metadata(
    title="Detect Faces in Video",
    description="Run standard face detection models hyperparallelized on video.",
    image=sieve.Image(
        url="https://developers.google.com/static/mediapipe/images/solutions/examples/face_detector.png"
    ),
    code_url="https://github.com/sieve-community/examples/tree/main/face_detection/wf.py",
    tags=["Detection", "Video"],
    readme=readme_content,
)

metadata = sieve.Metadata(
    description="Detect faces in an image with MediaPipe.",
    code_url="https://github.com/sieve-community/examples/tree/main/face_detection/wf.py",
    tags=["Detection", "Image"],
    readme=readme_content,
)


@sieve.Model(
    name="mediapipe-face-detector",
    gpu=False,
    python_version="3.8",
    python_packages=[
        "mediapipe==0.10.3",
        "opencv-python-headless==4.5.5.64",
    ],
    system_packages=["libgl1"],
    metadata=metadata,
)
class FaceDetector:
    def __setup__(self):
        import mediapipe as mp

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )

    def __predict__(self, img: sieve.Image) -> List:
        """
        :param img: Image to detect faces in
        :return: List of faces with their bounding boxes, classes, and scores
        """
        import cv2

        print("Starting prediction...")
        results = self.face_detection.process(
            cv2.cvtColor(img.array, cv2.COLOR_BGR2RGB)
        )
        outputs = []
        print("Finished prediction, outputting...")
        if results.detections:
            for detection in results.detections:
                ratio_box = [
                    detection.location_data.relative_bounding_box.xmin,
                    detection.location_data.relative_bounding_box.ymin,
                    detection.location_data.relative_bounding_box.width,
                    detection.location_data.relative_bounding_box.height,
                ]
                # box x1, y1, x2, y2
                box = [
                    int(ratio_box[0] * img.width),
                    int(ratio_box[1] * img.height),
                    int((ratio_box[0] + ratio_box[2]) * img.width),
                    int((ratio_box[1] + ratio_box[3]) * img.height),
                ]
                outputs.append(
                    {
                        "box": box,
                        "class_name": "face",
                        "score": detection.score[0],
                        "frame_number": None
                        if not hasattr(img, "frame_number")
                        else img.frame_number,
                    }
                )
        return outputs


@sieve.workflow(name="face-detection-image")
def mediapipe_face_detection(image: sieve.Image) -> List:
    """
    :param image: Image to detect faces in
    :return: List of faces with their bounding boxes, classes, and scores
    """
    return FaceDetector()(image)


@sieve.workflow(
    name="face-detection-video",
    metadata=wf_metadata,
)
def mediapipe_face_detection_vid(vid: sieve.Video) -> List:
    """
    :param vid: Video to detect faces in
    :return: List of faces with their bounding boxes, classes, scores, and frame numbers
    """
    video_splitter = sieve.reference("sieve/video-splitter")
    frames = video_splitter(vid)
    return FaceDetector()(frames)


if __name__ == "__main__":
    sieve.push(
        workflow="face-detection-image",
        inputs={
            "image": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/sama_avatar1.jpeg"
            }
        },
    )
