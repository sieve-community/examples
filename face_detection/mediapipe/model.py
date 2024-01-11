import sieve
from typing import List, Optional
from pydantic import BaseModel

readme_content = open("README.md", "r").read()
metadata = sieve.Metadata(
    description="Detect faces in image and video with MediaPipe.",
    code_url="https://github.com/sieve-community/examples/tree/main/face_detection/mediapipe",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/c4d968f5-f25a-412b-9102-5b6ab6dafcb4/19a6878d-e790-4e20-a4c9-99249b4499c2"
    ),
    tags=["Detection", "Image", "Video", "Face"],
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
        self.confidence_threshold = 0.5
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.confidence_threshold
        )

    def __predict__(
            self, 
            file: sieve.File,
            confidence_threshold: float = 0.5,
            start_frame: int = 0, 
            end_frame: int = -1, 
            fps: int = -1
        ) -> List[BoundingBox]:
        """
        :param file: Image or video file. If video, a generator is returned with the results for each frame.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param start_frame: The frame number to start processing from. Defaults to 0.
        :param end_frame: The frame number to stop processing at. Defaults to -1, which means the end of the video.
        :param fps: The fps to process the video at. Defaults to -1, which means the original fps of the video. If the specified fps is higher than the original fps, the original fps is used.
        :return: A dictionary with the results.
        """
        import cv2
        import os
        import time

        if confidence_threshold != self.confidence_threshold:
            self.confidence_threshold = confidence_threshold
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=self.confidence_threshold
            )

        video_extensions = ['mp4', 'avi', 'mov', 'flv']
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        file_extension = os.path.splitext(file.path)[1][1:]

        def process_frame(image_array, return_visualization=False):
            image_width, image_height = image_array.shape[1], image_array.shape[0]
            results = self.face_detection.process(
                cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            )
            outputs = []
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
                        int(ratio_box[0] * image_width),
                        int(ratio_box[1] * image_height),
                        int((ratio_box[0] + ratio_box[2]) * image_width),
                        int((ratio_box[1] + ratio_box[3]) * image_height),
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
            
            if not return_visualization:
                return outputs
            else:
                for output in outputs:
                    cv2.rectangle(
                        image_array,
                        (output["x1"], output["y1"]),
                        (output["x2"], output["y2"]),
                        (0, 255, 0),
                        4,  # Increased line thickness from 2 to 4
                    )
                return image_array, outputs
            
        if file_extension in video_extensions:
            video_path = file.path
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == -1 or fps > original_fps:
                fps = original_fps
                skip_frames = 1
            else:
                skip_frames = int(original_fps / fps)

            if end_frame == -1:
                end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            else:
                end_frame = min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)

            count = start_frame
            if count != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            start_time = time.time()
            while True:
                if skip_frames != 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                ret, frame = cap.read()
                if ret and count <= end_frame:
                    results = process_frame(frame)
                    if results:
                        yield {
                            "frame_number": count,
                            "boxes": results
                        }
                    else:
                        yield {
                            "frame_number": count,
                            "boxes": []
                        }
                    count += skip_frames
                else:
                    break

            cap.release()
            end_time = time.time()
            fps = (count / skip_frames) / (end_time - start_time)
            print(f"Processing FPS: {fps}")

        elif file_extension in image_extensions:
            image_path = file.path
            image_array = cv2.imread(image_path)
            results = process_frame(image_array)
            print(results)
            yield {
                "boxes": results,
                "frame_number": 0
            }
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions are {video_extensions + image_extensions}")

