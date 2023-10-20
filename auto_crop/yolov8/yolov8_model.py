import sieve

metadata = sieve.Metadata(
    description = "Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection and image segmentation model.",
    tags=["Video", "Detection", "Crop"],
    readme=open("yolo_README.md", "r").read()
)

@sieve.Model(
    name="yolov8",
    gpu=True,
    python_packages=[
        'ultralytics',
        'torch==1.13.1',
        'torchvision==0.14.1'
    ],
    cuda_version="11.7.1",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.10"
)
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO

        self.model = YOLO('yolov8l.pt')
    
    def __predict__(self, img: sieve.Image, classes: int = 0) -> dict:
        import numpy as np

        image_path = img.path
        results = self.model(image_path, classes=classes)
        results_dict = {
            "boxes": [],
        }

        for result in results:
            # Append boxes information to the dictionary
            box_info = {
                "xyxy": result.boxes.xyxy.cpu().numpy(),
                "xywh": result.boxes.xywh.cpu().numpy(),
                "xyxyn": result.boxes.xyxyn.cpu().numpy(),
                "xywhn": result.boxes.xywhn.cpu().numpy(),
                "conf": result.boxes.conf.cpu().numpy(),
                "cls": result.boxes.cls.cpu().numpy()
            }
            results_dict["boxes"].append(box_info)

        return results_dict