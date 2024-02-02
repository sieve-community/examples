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
    name="yolov8",
    gpu=True,
    python_packages=["ultralytics", "torch==1.13.1", "torchvision==0.14.1"],
    cuda_version="11.7.1",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.10",
    metadata=metadata,
    run_commands=[
        "mkdir -p /root/.models/",
        "wget -O /root/.models/yolov8l-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt",
        "wget -O /root/.models/yolov8n-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "pip install decord"
    ]
)
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO

        self.model = YOLO('yolov8l.pt')
        self.fast_model = YOLO('yolov8n.pt')
        self.pose_model = YOLO('yolov8l-pose.pt')
        self.face_model = YOLO("/root/.models/yolov8l-face.pt")
        self.face_fast_model = YOLO("/root/.models/yolov8n-face.pt")

    def __predict__(
            self,
            file: sieve.File,
            confidence_threshold: float = 0.5,
            classes: int = -1,
            models: str = "yolov8l",
            start_frame: int = 0,
            end_frame: int = -1,
            fps: int = -1,
            max_num_boxes: int = -1,
        ):
        """
        :param file: Image or video file.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param return_visualization: Whether to return the visualization of the results.
        :param classes: The class that should be detected. There are 80 classes to choose from for detections (see README). Entering -1 for the classes parameter detects all of them. To detect any one, you can enter the corresponding class number as input for classes.
        :param models: The models to use for inference. The models are specified as a comma-separated string. The supported models are yolov8l, yolov8n, yolov8l-pose, yolov8l-face, and yolov8n-face. The default model is yolov8l. If multiple models are specified, the results from all the models are combined.
        :param start_frame: The frame number to start processing from. Defaults to 0.
        :param end_frame: The frame number to stop processing at. Defaults to -1, which means the end of the video.
        :param speed_boost: Whether to use the faster version of YOLOv8. This is less accurate but faster.
        :param fps: The fps to process the video at. Defaults to -1, which means the original fps of the video. If the specified fps is higher than the original fps, the original fps is used.
        :param max_num_boxes: The maximum number of boxes to return per frame. Defaults to -1, which means all boxes are returned. Otherwise, the boxes are sorted by confidence and the top max_num_boxes are returned.
        :return: A dictionary with the results.
        """
        print("Got request...")
        import cv2
        import os
        import time
        import torch

        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise Exception("No CUDA devices are visible. Please check your CUDA setup.")

        video_extensions = ["mp4", "avi", "mov", "flv"]
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

        # split the models string by comma and remove any whitespace
        models = [model.strip() for model in models.split(",")]
        models_to_use = []
        # remove any duplicate models
        models = list(set(models))
        for model in models:
            # strip the model string of any whitespace
            model = model.strip()
            if model == "yolov8l":
                models_to_use.append(self.model)
            elif model == "yolov8n":
                models_to_use.append(self.fast_model)
            elif model == "yolov8l-pose":
                models_to_use.append(self.pose_model)
            elif model == "yolov8l-face":
                models_to_use.append(self.face_model)
            elif model == "yolov8n-face":
                models_to_use.append(self.face_fast_model)
            else:
                raise ValueError(
                    f"Unsupported model: {model}. Supported models are yolov8l, yolov8n, yolov8l-pose, yolov8l-face, and yolov8n-face."
                )

        file_extension = os.path.splitext(file.path)[1][1:]
        print("Starting inference...")
        if file_extension in video_extensions:
            video_path = file.path
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == -1 or fps > original_fps:
                fps = original_fps
                skip_frames = 1
            else:
                skip_frames = int(original_fps / fps)

            if end_frame == -1:
                end_frame = int(num_frames) - 1
            else:
                end_frame = min(end_frame, int(num_frames) - 1)

            count = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            num_frames_to_process = int((end_frame - start_frame) / skip_frames) + 1
            start_time = time.time()
            outputs = []
            while True:
                if skip_frames != 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                ret, frame = cap.read()
                if ret and count <= end_frame:
                    combined_boxes = []
                    for x, model_to_use in enumerate(models_to_use):
                        if classes == -1:
                            results = model_to_use(frame)
                        else:
                            results = model_to_use(frame, classes=classes)
                        face_detection = "face" in models[x]
                        results_dict = self.__process_results__(results, face_detection=face_detection)
                        # Filter results based on confidence threshold
                        results_dict["boxes"] = [box for box in results_dict["boxes"] if box["confidence"] > confidence_threshold]
                        combined_boxes.extend(results_dict["boxes"])
                    
                    if max_num_boxes != -1 and len(combined_boxes) > max_num_boxes:
                        combined_boxes = sorted(combined_boxes, key=lambda x: x["confidence"], reverse=True)[:max_num_boxes]
                    
                    output_dict = {"frame_number": count, "boxes": combined_boxes}
                    outputs.append(output_dict)
                    count += skip_frames
                else:
                    break
                if count % 100 == 0:
                    print(f"Processed {round((len(outputs) / num_frames_to_process * 100), 2)}% of frames ({len(outputs)} / {num_frames_to_process})")

            # del vr
            cap.release()
            end_time = time.time()
            fps = (count / skip_frames) / (end_time - start_time)
            print(f"Processing FPS: {fps}")
            return outputs
        elif file_extension in image_extensions:
            image_path = file.path
            combined_boxes = []
            for model_to_use in models_to_use:
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
                combined_boxes.extend(results_dict["boxes"])

            return {"boxes": combined_boxes}
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
                    "x1": int(box.xyxy.cpu().numpy().tolist()[0][0]),
                    "y1": int(box.xyxy.cpu().numpy().tolist()[0][1]),
                    "x2": int(box.xyxy.cpu().numpy().tolist()[0][2]),
                    "y2": int(box.xyxy.cpu().numpy().tolist()[0][3]),
                    "width": int(box.xyxy.cpu().numpy().tolist()[0][2]
                    - box.xyxy.cpu().numpy().tolist()[0][0]),
                    "height": int(box.xyxy.cpu().numpy().tolist()[0][3]
                    - box.xyxy.cpu().numpy().tolist()[0][1]),
                    "confidence": float(box.conf.cpu().numpy().tolist()[0]),
                    "class_number": int(box.cls.cpu().numpy().tolist()[0]),
                }
                box_info["class_name"] = self.model.names[box_info["class_number"]] if not face_detection else "face"
                results_dict["boxes"].append(box_info)

        return results_dict
