import os
import sieve

from config import CACHE


def get_object_bbox(image: sieve.File, object_name: str):
    if CACHE and os.path.exists("bbox.txt"):
        with open("bbox.txt", "r") as f:
            return list(map(int, f.read().split(',')))

    yolo = sieve.function.get('sieve/yolov8')

    response = yolo.run(
        file=image,
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]
    bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]

    if CACHE:
        with open("bbox.txt", "w") as f:
            f.write(','.join(map(str, bounding_box)))

    return bounding_box
