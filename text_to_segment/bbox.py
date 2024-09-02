import sieve

def get_object_bbox(image: sieve.File, object_name: str):
    yolo = sieve.function.get('sieve/yolov8')

    response = yolo.run(
        file=image,
        classes=object_name,
        models='yolov8l-world',
    )

    box = response['boxes'][0]
    bounding_box = [box['x1'],box['y1'],box['x2'],box['y2']]

    return bounding_box
