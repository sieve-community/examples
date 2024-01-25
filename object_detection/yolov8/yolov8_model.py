import sieve

metadata = sieve.Metadata(
    description="Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection model.",
    code_url="https://github.com/sieve-community/examples/blob/main/object_detection/yolov8",
    tags=["Image", "Object", "Detection"],
    image=sieve.Image(
        url="https://www.freecodecamp.org/news/content/images/2023/04/compvision_tasks.png"
    ),
    readme=open("README.md", "r").read(),
)


@sieve.Model(
    name="yolov8l",
    gpu=True,
    python_packages=["ultralytics", "torch==1.13.1", "torchvision==0.14.1"],
    cuda_version="11.7.1",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.10",
    metadata=metadata,
    run_commands=[
        "mkdir -p /root/.models/",
        "wget -O /root/.models/yolov8l-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt"
    ]
)
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO

        self.model = YOLO('yolov8l.pt')
        self.face_model = YOLO("/root/.models/yolov8l-face.pt")

    def __predict__(
            self,
            file: sieve.File,
            confidence_threshold: float = 0.5,
            classes: int = -1,
            face_detection: bool = False,
            start_frame: int = 0,
            end_frame: int = -1,
            fps: int = -1,
        ):
        """
        :param file: Image or video file. If video, a generator is returned with the results for each frame.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param return_visualization: Whether to return the visualization of the results.
        :param classes: The class that should be detected. There are 80 classes to choose from for detections (see README). Entering -1 for the classes parameter detects all of them. To detect any one, you can enter the corresponding class number as input for classes.
        :param face_detection: Whether to use the finetuned face detection only version of YOLOv8.
        :param start_frame: The frame number to start processing from. Defaults to 0.
        :param end_frame: The frame number to stop processing at. Defaults to -1, which means the end of the video.
        :param fps: The fps to process the video at. Defaults to -1, which means the original fps of the video. If the specified fps is higher than the original fps, the original fps is used.
        :return: A dictionary with the results.
        """
        import cv2
        import os
        import time
        import torch

        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise Exception("No CUDA devices are visible. Please check your CUDA setup.")

        video_extensions = ["mp4", "avi", "mov", "flv"]
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

        model_to_use = self.model if not face_detection else self.face_model

        file_extension = os.path.splitext(file.path)[1][1:]
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            start_time = time.time()
            while True:
                if skip_frames != 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                ret, frame = cap.read()
                if ret and count <= end_frame:
                    if classes == -1:
                        results = model_to_use(frame)
                    else:
                        results = model_to_use(frame, classes=classes)
                    results_dict = self.__process_results__(results, face_detection=face_detection)
                    results_dict["frame_number"] = count
                    count += skip_frames
                    # Filter results based on confidence threshold
                    results_dict["boxes"] = [box for box in results_dict["boxes"] if box["confidence"] > confidence_threshold]
                    yield results_dict
                else:
                    break

            cap.release()
            end_time = time.time()
            fps = (count / skip_frames) / (end_time - start_time)
            print(f"Processing FPS: {fps}")
        elif file_extension in image_extensions:
            image_path = file.path
            if classes == -1:
                results = model_to_use(image_path)
            else:
                results = model_to_use(image_path, classes=classes)
            results_dict = self.__process_results__(results, face_detection=face_detection)

            # Filter results based on confidence threshold
            results_dict["boxes"] = [
                box
                for box in results_dict["boxes"]
                if box["confidence"] > confidence_threshold
            ]
            if results_dict["boxes"]:
                yield results_dict
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Supported extensions are {video_extensions + image_extensions}"
            )

    def __process_results__(self, results, face_detection=False) -> dict:
        results_dict = {
            "boxes": [],
        }

        for result in results:
            # Append boxes information to the dictionary
            for box in result.boxes:
                box_info = {
                    "x1": box.xyxy.cpu().numpy().tolist()[0][0],
                    "y1": box.xyxy.cpu().numpy().tolist()[0][1],
                    "x2": box.xyxy.cpu().numpy().tolist()[0][2],
                    "y2": box.xyxy.cpu().numpy().tolist()[0][3],
                    "width": box.xyxy.cpu().numpy().tolist()[0][2]
                    - box.xyxy.cpu().numpy().tolist()[0][0],
                    "height": box.xyxy.cpu().numpy().tolist()[0][3]
                    - box.xyxy.cpu().numpy().tolist()[0][1],
                    "confidence": box.conf.cpu().numpy().tolist()[0],
                    "class_number": box.cls.cpu().numpy().tolist()[0],
                }
                box_info["class_name"] = self.model.names[box_info["class_number"]] if not face_detection else "face"
                results_dict["boxes"].append(box_info)

        return results_dict
