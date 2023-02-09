import sieve
import torch
import cv2
from detect import load_model, predict

@sieve.Model(
    name="u2net",
    gpu = True,
    python_packages=[
        "six==1.16.0",
        "datetime==4.7",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.8.1",
        "torchvision==0.9.1"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/lip/models/",
        "wget -c 'https://storage.googleapis.com/sieve-public-model-assets/bg-removal/u2netp.pth' -P /root/.cache/lip/models/"
    ],
    persist_output=True
)
class U2Net:
    def __setup__(self):
        self.model = load_model()

    def __predict__(self, img: sieve.Image) -> sieve.Image:
        frame_data = cv2.cvtColor(img.array, cv2.COLOR_BGR2RGB)
        width = frame_data.shape[1]
        height = frame_data.shape[0]
        
        output_image = predict(self.model, frame_data)
        # resize to original size
        output_image = cv2.resize(output_image, (width, height), interpolation=cv2.INTER_CUBIC)
        if hasattr(img, "frame_number"):
            return sieve.Image(array=output_image, frame_number=img.frame_number)
        else:
            return sieve.Image(array=output_image)