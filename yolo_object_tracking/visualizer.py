import sieve
from typing import List, Dict


@sieve.function(
    name="draw-boxes",
    gpu=False,
    python_version="3.8",
    iterator_input=True,
    python_packages=["uuid==1.30"],
)
def draw_boxes(images: sieve.Image, boxes: List) -> sieve.Image:
    """
    :param images: Source image to draw boxes on
    :param boxes: List of objects with their bounding boxes, classes, and scores
    :return: Image with bounding boxes drawn
    """
    import numpy as np
    import cv2
    import uuid

    image_paths = []
    for im in images:
        image_paths.append(im)

    l = []
    for i in boxes:
        if len(i) > 0:
            l.append(i)

    boxes_by_frame_number = sorted(l, key=lambda k: k[0]["frame_number"])
    images_by_frame_number = sorted(image_paths, key=lambda k: k.frame_number)

    for i in range(len(boxes_by_frame_number)):
        boxes = boxes_by_frame_number[i]
        image = images_by_frame_number[i]
        img = cv2.imread(image.path)
        for box in boxes:
            box["box"] = np.array(box["box"]).astype(int)
            color = (255, 64, 114)
            color_lighter = (255, 128, 128)
            cv2.rectangle(
                img,
                (box["box"][0], box["box"][1]),
                (box["box"][2], box["box"][3]),
                color,
                2,
            )
            cv2.putText(
                img,
                box["class_name"],
                (box["box"][0], box["box"][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_lighter,
                2,
            )

        new_path = f"{uuid.uuid4()}.jpg"
        cv2.imwrite(new_path, img)
        yield sieve.Image(
            path=new_path, frame_number=boxes[0]["frame_number"], fps=image.fps
        )
