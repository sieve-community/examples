import sieve

metadata = sieve.Metadata(
    description="YOLOv8 real-time object detection model with COCO, face, and world variants.",
    code_url="https://github.com/sieve-community/examples/blob/main/object_detection/yolov8",
    tags=["Image", "Object", "Detection"],
    image=sieve.Image(
        url="https://www.freecodecamp.org/news/content/images/2023/04/compvision_tasks.png"
    ),
    readme=open("README.md", "r").read(),
)


@sieve.Model(
    name="yolov8",
    gpu=sieve.gpu.T4(split=2),
    python_packages=["ultralytics", "torch==1.13.1", "torchvision==0.14.1"],
    cuda_version="11.7.1",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.10",
    metadata=metadata,
    run_commands=[
        "mkdir -p /root/.models/",
        "wget -O /root/.models/yolov8l-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt",
        "wget -O /root/.models/yolov8n-face.pt https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
        "pip install decord",
        "pip install 'imageio[ffmpeg]'",
        "pip install git+https://github.com/ultralytics/ultralytics.git@29dc1a3987eb8aa2d55d067daffdd26d14929020",
    ]
)
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO

        self.model = YOLO('yolov8l.pt')
        self.fast_model = YOLO('yolov8n.pt')
        self.face_model = YOLO("/root/.models/yolov8l-face.pt")
        self.face_fast_model = self.face_model
        self.world_model = YOLO('yolov8l-worldv2.pt')
        self.world_fast_model = YOLO('yolov8s-worldv2.pt')
        self.current_world_classes = None
        self.current_world_fast_classes = None

    def __predict__(
            self,
            file: sieve.File,
            confidence_threshold: float = 0.05,
            classes: str = "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush",
            models: str = "yolov8l",
            start_frame: int = 0,
            end_frame: int = -1,
            fps: float = -1,
            max_num_boxes: int = -1,
        ):
        """
        :param file: Image or video file.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param return_visualization: Whether to return the visualization of the results.
        :param classes: The classes to use for inference. The classes are specified as a comma-separated string. Only applicable if the model is yolov8l-world or yolov8s-world which support natural language prompts. The default classes are the COCO classes.
        :param models: The models to use for inference. The models are specified as a comma-separated string. The supported models are yolov8l, yolov8n, yolov8l-face, yolov8n-face, yolov8l-world, and yolov8s-world. The default model is yolov8l. If multiple models are specified, the results from all the models are combined.
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

        def process_categories(categories):
            categories = categories.split(",")
            categories = [category.strip() for category in categories]
            return categories + [" "]
        # remove any duplicate models
        models = list(set(models))
        for model in models:
            # strip the model string of any whitespace
            model = model.strip()
            if model == "yolov8l":
                models_to_use.append(self.model)
            elif model == "yolov8n":
                models_to_use.append(self.fast_model)
            elif model == "yolov8l-face":
                models_to_use.append(self.face_model)
            elif model == "yolov8n-face":
                models_to_use.append(self.face_fast_model)
            elif model == "yolov8l-world":
                models_to_use.append(self.world_model)
                new_classes = process_categories(classes)
                if self.current_world_classes != new_classes:
                    self.current_world_classes = new_classes
                    self.world_model.set_classes(new_classes)
            elif model == "yolov8s-world":
                models_to_use.append(self.world_fast_model)
                new_classes = process_categories(classes)
                if self.current_world_fast_classes != new_classes:
                    self.current_world_fast_classes = new_classes
                    self.world_fast_model.set_classes(new_classes)
            else:
                raise ValueError(
                    f"Unsupported model: {model}. Supported models are yolov8l, yolov8n, yolov8l-face, yolov8n-face, yolov8l-world, and yolov8s-world"
                )

        file_extension = os.path.splitext(file.path)[1][1:]
        print("Starting inference...")
        if file_extension in video_extensions:
            video_path = file.path
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps == -1 or fps > original_fps:
                fps = original_fps
                
            if end_frame == -1:
                end_frame = int(num_frames) - 1
            else:
                end_frame = min(end_frame, int(num_frames) - 1)

            start_time = time.time()
            outputs = []

            frames_number_to_read = []
            for i in range(int(end_frame - start_frame) + 1):
                frame_number = int(start_frame + i * (original_fps / fps))
                if start_frame <= frame_number < end_frame:
                    frames_number_to_read.append(frame_number)

            if end_frame not in frames_number_to_read:
                frames_number_to_read.append(end_frame)
            
            t = time.time()
            import imageio
            cap = imageio.get_reader(file.path)
            current_frame_number = start_frame
            try:
                if start_frame != 0:
                    cap.set_image_index(start_frame)
                print(f"load & seek time: {round(time.time() - t, 2)}s")
                current_frame = cap.get_next_data()
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            except IndexError:
                return [{"frame_number": current_frame_number, "boxes": []}]
            for p in frames_number_to_read:
                frame_to_process = None
                if p == current_frame_number:
                    frame_to_process = current_frame
                    # convert the frame to RGB
                    frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
                else:
                    new_frame_number = p
                    if new_frame_number != current_frame_number + 1:
                        # print(f"Seeking to frame {new_frame_number}")
                        cap.set_image_index(new_frame_number)
                    try:
                        new_frame = cap.get_next_data()
                    except IndexError:
                        break
                    if new_frame is not None:
                        current_frame_number = p
                        current_frame = new_frame
                        frame_to_process = current_frame
                        # frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
                    else:
                        break

                if frame_to_process is not None:
                    combined_boxes = []
                    for x, model_to_use in enumerate(models_to_use):
                        results = model_to_use.predict(frame_to_process, conf=confidence_threshold)
                        results_dict = self.__process_results__(results, model_to_use)
                        # Filter results based on confidence threshold
                        results_dict["boxes"] = [box for box in results_dict["boxes"] if box["confidence"] > confidence_threshold]
                        combined_boxes.extend(results_dict["boxes"])
                    
                    if max_num_boxes != -1 and len(combined_boxes) > max_num_boxes:
                        combined_boxes = sorted(combined_boxes, key=lambda x: x["confidence"], reverse=True)[:max_num_boxes]
                    
                    output_dict = {"frame_number": p, "boxes": combined_boxes}
                    outputs.append(output_dict)

                if len(outputs) % 100 == 0:
                    print(f"Processed {round((len(outputs) / len(frames_number_to_read) * 100), 2)}% of frames ({len(outputs)} / {len(frames_number_to_read)})")
            cap.close()
            end_time = time.time()
            fps = len(outputs) / (end_time - start_time)
            print(f"Processing FPS: {fps}")
            return outputs
        elif file_extension in image_extensions:
            image_path = file.path
            combined_boxes = []
            for x, model_to_use in enumerate(models_to_use):
                results = model_to_use.predict(image_path, conf=confidence_threshold)
                results_dict = self.__process_results__(results, model_to_use)

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

    def __process_results__(self, results, model_to_use) -> dict:
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
                box_info["class_name"] = model_to_use.names[box_info["class_number"]]
                results_dict["boxes"].append(box_info)

        return results_dict
