# yolov8l

Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection model.

![example image](https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png)

Check out the original repo [here](https://github.com/ultralytics/ultralytics).

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
