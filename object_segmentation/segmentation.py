import sieve


@sieve.Model(
    name="instance-segmentation",
    gpu=True,
    python_packages=[
        "opencv-python==4.6.0.66",
        "numpy==1.24.2",
        "torch==1.9.0",
        "torchvision==0.10.0",
        "pixellib==0.7.1",
        "pycocotools==2.0.2",
    ],
    run_commands=[
        "mkdir -p /root/.cache/pointrend/models/",
        "wget https://storage.googleapis.com/mango-public-models/pointrend_resnet50.pkl -P /root/.cache/pointrend/models"
    ]
)
class InstanceSegmentation:
    def __setup__(self):
        import time
        from pixellib.torchbackend.instance import instanceSegmentation

        self.model = instanceSegmentation()
        self.model.load_model("/root/.cache/pointrend/models/pointrend_resnet50.pkl")
        self.class_names = self.model.class_names
        self.class_colors = list(self.getBrightDistinctColors(len(self.class_names)))

    def HSVToRGB(self, h, s, v):
        import colorsys

        (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def getDistinctColors(self, n):
        huePartition = 1.0 / (n + 1)
        return (self.HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

    def getBrightDistinctColors(self, n):
        huePartition = 1.0 / (n + 1)
        return (self.HSVToRGB(huePartition * value, 1.0, 0.5) for value in range(0, n))

    def __predict__(self, image: sieve.Image) -> sieve.Image:
        '''
        :param image: Image to run segmentation on
        :return: Image with segmentation mask overlayed
        '''
        image_copy = image.array.copy()
        import cv2
        import numpy as np

        outputs = self.model.predictor.segment(
            cv2.cvtColor(image.array, cv2.COLOR_BGR2RGB)
        )
        masks = outputs["instances"].pred_masks
        class_ids = outputs["instances"].pred_classes
        scores = outputs["instances"].scores
        boxes = outputs["instances"].pred_boxes.tensor
        # Draw masks on the image
        init_mask = np.zeros_like(image_copy)
        for i in range(len(masks)):
            mask = masks[i].cpu().numpy()
            class_id = class_ids[i]
            score = scores[i]
            box = boxes[i].cpu().numpy()
            color = self.class_colors[class_id]
            self.draw_mask(init_mask, mask, color, score, box)
        output_image = self.blend_mask_image(image_copy, init_mask, color, score, box)
        if hasattr(image, "fps") and hasattr(image, "frame_number"):
            return sieve.Image(array=output_image, fps=image.fps, frame_number=image.frame_number)
        if hasattr(image, "fps"):
            return sieve.Image(array=output_image, fps=image.fps)
        if hasattr(image, "frame_number"):
            return sieve.Image(array=output_image, frame_number=image.frame_number)
        return sieve.Image(array=output_image)

    def draw_mask(self, image, mask, color, score, box):
        image[mask] = color

    def blend_mask_image(self, image, mask, color, score, box):
        import cv2

        alpha = 0.5
        beta = 1.0 - alpha
        gamma = 0.0
        return cv2.addWeighted(image, alpha, mask, beta, gamma, image)
