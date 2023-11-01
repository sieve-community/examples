import sieve
from typing import List, Optional
from pydantic import BaseModel

readme_content = open("README.md", "r").read()
metadata = sieve.Metadata(
    description="Detect faces in an image with MediaPipe.",
    code_url="https://github.com/sieve-community/examples/tree/main/face_detection/mediapipe",
    image=sieve.Image(
        url="https://yt3.googleusercontent.com/ytc/AOPolaRCLtk0dFe9XgCDF3JAhadHDajjlo85lw0gf88O=s900-c-k-c0x00ffffff-no-rj"
    ),
    tags=["Detection", "Image", "Face"],
    readme=readme_content,
)

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: Optional[str] = None
    score: Optional[float] = None

@sieve.Model(
    name="mediapipe_face_detector",
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

    def __predict__(self, image: sieve.Image) -> List[BoundingBox]:
        """
        :param image: Image to detect faces in
        :return: List of faces with their bounding boxes, classes, and scores
        """
        import cv2

        print("Starting prediction...")
        results = self.face_detection.process(
            cv2.cvtColor(image.array, cv2.COLOR_BGR2RGB)
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
                    int(ratio_box[0] * image.width),
                    int(ratio_box[1] * image.height),
                    int((ratio_box[0] + ratio_box[2]) * image.width),
                    int((ratio_box[1] + ratio_box[3]) * image.height),
                ]
                outputs.append(
                    BoundingBox(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        class_name="face",
                        score=float(detection.score[0]),
                    ).dict()
                )
        
        for output in outputs:
            cv2.rectangle(
                image.array,
                (output["x1"], output["y1"]),
                (output["x2"], output["y2"]),
                (0, 255, 0),
                4,  # Increased line thickness from 2 to 4
            )
        yield sieve.Image(array=image.array)
        yield outputs
