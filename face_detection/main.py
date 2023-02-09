import sieve
from typing import Dict, List
import mediapipe as mp
import cv2

@sieve.Model(
    name="mediapipe-face-detector",
    gpu = False,
    python_version="3.8",
    python_packages=[
        'mediapipe==0.9.0'
    ]
)
class FaceDetector:
    def __setup__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def __predict__(self, img: sieve.Image) -> List:
        results = self.face_detection.process(cv2.cvtColor(img.array, cv2.COLOR_BGR2RGB))
        outputs = []
        if results.detections:
            for detection in results.detections:
                outputs.append({
                    "box": [detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height],
                    "class_name": "face",
                    "score": detection.score[0],
                    "frame_number": None if not hasattr(img, "frame_number") else img.frame_number
                })
        return outputs

@sieve.workflow(name="mediapipe-face-detection")
def mediapipe_face_detection(image: sieve.Image) -> List:
    return FaceDetector()(image)
