import sieve
import torch
import numpy as np
import cv2


def blur_img(img, factor=20):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)

    # ensure the shape of the kernel is odd
    if kW % 2 == 0:
        kW = kW - 1
    if kH % 2 == 0:
        kH = kH - 1

    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img


def soft_blur_with_mask(image: np.ndarray, mask: np.ndarray, strength=10) -> np.ndarray:
    blurred_img = blur_img(image, factor=strength)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask / 255
    return (image * mask + blurred_img * (1 - mask)).astype(np.uint8)


@sieve.Model(
    name="u2netp_mask",
    gpu=True,
    python_packages=[
        "six==1.16.0",
        "datetime==4.7",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.8.1",
        "torchvision==0.9.1",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/lip/models/",
        "wget -c 'https://storage.googleapis.com/sieve-public-model-assets/bg-removal/u2netp.pth' -P /root/.cache/lip/models/",
    ],
    persist_output=True,
)
class U2NetMask:
    def __setup__(self):
        from detect import load_model

        self.model = load_model()

    def __predict__(self, img: sieve.Image) -> sieve.Image:
        from detect import predict

        frame_data = cv2.cvtColor(img.array, cv2.COLOR_BGR2RGB)
        width = frame_data.shape[1]
        height = frame_data.shape[0]

        output_image = predict(self.model, frame_data)
        # resize to original size
        output_image = cv2.resize(
            output_image, (width, height), interpolation=cv2.INTER_CUBIC
        )
        if hasattr(img, "fps") and hasattr(img, "frame_number"):
            return sieve.Image(
                array=output_image, fps=img.fps, frame_number=img.frame_number
            )
        if hasattr(img, "fps"):
            return sieve.Image(array=output_image, fps=img.fps)
        if hasattr(img, "frame_number"):
            return sieve.Image(array=output_image, frame_number=img.frame_number)
        else:
            return sieve.Image(array=output_image)


@sieve.Model(
    name="u2netp_blur",
    gpu=True,
    python_packages=[
        "six==1.16.0",
        "datetime==4.7",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.8.1",
        "torchvision==0.9.1",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/lip/models/",
        "wget -c 'https://storage.googleapis.com/sieve-public-model-assets/bg-removal/u2netp.pth' -P /root/.cache/lip/models/",
    ],
    persist_output=True,
)
class U2NetBlur:
    def __setup__(self):
        from detect import load_model

        self.model = load_model()

    def __predict__(self, img: sieve.Image) -> sieve.Image:
        from detect import predict

        frame_data = cv2.cvtColor(img.array, cv2.COLOR_BGR2RGB)
        width = frame_data.shape[1]
        height = frame_data.shape[0]

        output_image = predict(self.model, frame_data)
        # resize to original size
        output_image = cv2.resize(
            output_image, (width, height), interpolation=cv2.INTER_CUBIC
        )

        output_image = soft_blur_with_mask(frame_data, output_image, strength=10)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if hasattr(img, "fps") and hasattr(img, "frame_number"):
            return sieve.Image(
                array=output_image, fps=img.fps, frame_number=img.frame_number
            )
        if hasattr(img, "fps"):
            return sieve.Image(array=output_image, fps=img.fps)
        if hasattr(img, "frame_number"):
            return sieve.Image(array=output_image, frame_number=img.frame_number)
        else:
            return sieve.Image(array=output_image)
