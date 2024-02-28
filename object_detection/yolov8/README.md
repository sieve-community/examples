# yolov8

Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection model.

![example image](https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png)

Check out the original repo [here](https://github.com/ultralytics/ultralytics).

## Supported Models
We support the following YOLOv8 models:
- `yolov8l`: YOLOv8-Large
- `yolov8n`: YOLOv8-Nano
- `yolov8l-face`: YOLOv8-Large trained on the WIDER FACE dataset
- `yolov8n-face`: YOLOv8-Nano trained on the WIDER FACE dataset
- `yolov8l-world`: YOLOv8-Large used for open-vocabulary object detection (specify any object you want to detect)
- `yolov8s-world`: YOLOv8-Small used for open-vocabulary object detection (specify any object you want to detect)

## Output Format

The model returns a dictionary with the following structure:

```json
{
  "boxes": [
    {
      "x1": 281.36834716796875,
      "y1": 114.6531982421875,
      "x2": 937.3942260742188,
      "y2": 712.1339111328125,
      "width": 656.02587890625,
      "height": 597.480712890625,
      "confidence": 0.9540532827377319,
      "class_number": 0,
      "class_name": "person"
    },
    {
      "x1": 932.1299438476562,
      "y1": 610.4577026367188,
      "x2": 1279.69775390625,
      "y2": 711.7876586914062,
      "width": 347.56781005859375,
      "height": 101.3299560546875,
      "confidence": 0.6923111081123352,
      "class_number": 56,
      "class_name": "chair"
    }
  ],
  "frame_number": 0
}
```

Each box in the "boxes" array represents a detected object. The fields "x1", "y1", "x2", "y2" represent the coordinates of the top left and bottom right corners of the bounding box respectively. The "width" and "height" fields represent the width and height of the bounding box. The "confidence" field represents the confidence score of the detection. The "class_number" and "class_name" fields represent the class of the detected object.

## Classes
If you are using the `yolov8l-world` or `yolov8s-world` models, you can specify any object you want to detect using the `classes` parameter.

Otherwise, there are 80 classes to choose from for detections. These are the possible detection classes:
```python
"classes": {
  0: "person",
  1: "bicycle",
  2: "car",
  3: "motorcycle",
  4: "airplane",
  5: "bus",
  6: "train",
  7: "truck",
  8: "boat",
  9: "traffic light",
  10: "fire hydrant",
  11: "stop sign",
  12: "parking meter",
  13: "bench",
  14: "bird",
  15: "cat",
  16: "dog",
  17: "horse",
  18: "sheep",
  19: "cow",
  20: "elephant",
  21: "bear",
  22: "zebra",
  23: "giraffe",
  24: "backpack",
  25: "umbrella",
  26: "handbag",
  27: "tie",
  28: "suitcase",
  29: "frisbee",
  30: "skis",
  31: "snowboard",
  32: "sports ball",
  33: "kite",
  34: "baseball bat",
  35: "baseball glove",
  36: "skateboard",
  37: "surfboard",
  38: "tennis racket",
  39: "bottle",
  40: "wine glass",
  41: "cup",
  42: "fork",
  43: "knife",
  44: "spoon",
  45: "bowl",
  46: "banana",
  47: "apple",
  48: "sandwich",
  49: "orange",
  50: "broccoli",
  51: "carrot",
  52: "hot dog",
  53: "pizza",
  54: "donut",
  55: "cake",
  56: "chair",
  57: "couch",
  58: "potted plant",
  59: "bed",
  60: "dining table",
  61: "toilet",
  62: "tv",
  63: "laptop",
  64: "mouse",
  65: "remote",
  66: "keyboard",
  67: "cell phone",
  68: "microwave",
  69: "oven",
  70: "toaster",
  71: "sink",
  72: "refrigerator",
  73: "book",
  74: "clock",
  75: "vase",
  76: "scissors",
  77: "teddy bear",
  78: "hair drier",
  79: "toothbrush"
}
```