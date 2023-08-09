import sieve
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


@sieve.Model(
    name="segment-anything-point",
    gpu=True,
    python_version="3.9",
    python_packages=[
        "opencv-python==4.5.5.62",
        "pycocotools==2.0.2",
        "matplotlib==3.5.1",
        "onnxruntime==1.9.0",
        "onnx==1.10.2",
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libavcodec58",
        "ffmpeg",
    ],
    run_commands=[
        "mkdir -p /root/.cache/sam/models/",
        "wget -c 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' -P /root/.cache/sam/models/",
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
    ],
    iterator_input=True,
)
class SegmentAnything:
    def __setup__(self):
        sam_checkpoint = "/root/.cache/sam/models/sam_vit_h_4b8939.pth"
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

    def __predict__(self, img: sieve.Image, x: int, y: int) -> sieve.Image:
        """
        :param img: Image to run segmentation on
        :param x: X coordinate to center segmentation on
        :param y: Y coordinate to center segmentation on
        :return: Image with segmentation mask
        """
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image

        img, x, y = list(img)[0], list(x)[0], list(y)[0]
        img = cv2.imread(img.path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        color = np.array([255 / 255, 255 / 255, 255 / 255, 0.6])
        for i, (mask, score) in enumerate(zip(masks, scores)):
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            im = Image.fromarray(mask_image)
            im.save(f"{i}.png")
            yield sieve.Image(path=f"{i}.png")


@sieve.workflow(name="object-segmentation-by-point")
def object_segmentation_by_point(img: sieve.Image, x: int, y: int) -> sieve.Image:
    """
    :param img: Image to run segmentation on
    :param x: X coordinate to center segmentation on
    :param y: Y coordinate to center segmentation on
    :return: Image with segmentation mask
    """

    return SegmentAnything()(img, x, y)


if __name__ == "__main__":
    sieve.push(
        object_segmentation_by_point,
        inputs={
            "img": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/sama_avatar1.jpeg"
            },
            "x": 100,
            "y": 100,
        },
    )
