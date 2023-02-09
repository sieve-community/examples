import sieve
from typing import List, Dict

@sieve.Model(
    name="yolo",
    gpu = True,
    python_packages=[
        "torch==1.8.1",
        "pandas==1.5.2",
        "opencv-python-headless==4.5.4.60",
        "ipython==8.4.0",
        "torch==1.8.1",
        "torchvision==0.9.1",
        "psutil==5.8.0",
        "seaborn==0.11.2"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8"
)
class Yolo:
    def __setup__(self):
        import torch
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

    def __predict__(self, img: sieve.Image) -> List:
        results = self.yolo_model(img.array)
        outputs = []
        for pred in reversed(results.pred):
            for *box, conf, cls in reversed(pred):
                cls_name = results.names[int(cls)]
                box = [float(i) for i in box]
                score = float(conf)
                outputs.append({
                    "box": box,
                    "class_name": cls_name,
                    "score": score,
                    "frame_number": None if not hasattr(img, "frame_number") else img.frame_number
                })
        return outputs
