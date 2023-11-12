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
)
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolov8l.pt")

    def __predict__(
        self, file: sieve.File, confidence_threshold: float = 0.5, classes: int = -1
    ):
        """
        :param file: Image or video file. If video, a generator is returned with the results for each frame.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param return_visualization: Whether to return the visualization of the results.
        :param classes: The class (more info in README) that should be detected.
        :return: A dictionary with the results.
        """
        import cv2
        import os
        import time

        video_extensions = ["mp4", "avi", "mov", "flv"]
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

        file_extension = os.path.splitext(file.path)[1][1:]
        if file_extension in video_extensions:
            video_path = file.path
            cap = cv2.VideoCapture(video_path)
            count = 0
            start_time = time.time()
            while True:
                ret, frame = cap.read()
                if ret:
                    if classes == -1:
                        results = self.model(frame)
                    else:
                        results = self.model(frame, classes=classes)
                    results_dict = self.__process_results__(results)
                    results_dict["frame_number"] = count
                    count += 1
                    # Filter results based on confidence threshold
                    results_dict["boxes"] = [
                        box
                        for box in results_dict["boxes"]
                        if box["confidence"] > confidence_threshold
                    ]
                    yield results_dict
                else:
                    break

            cap.release()
            end_time = time.time()
            fps = count / (end_time - start_time)
            print(f"Processing FPS: {fps}")
        elif file_extension in image_extensions:
            image_path = file.path
            if classes == -1:
                results = self.model(image_path)
            else:
                results = self.model(image_path, classes=classes)
            results_dict = self.__process_results__(results)

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

    def __process_results__(self, results) -> dict:
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
                box_info["class_name"] = self.model.names[box_info["class_number"]]
                results_dict["boxes"].append(box_info)

        return results_dict
