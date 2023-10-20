# Yolov8

 Ultralytics YOLOv8, the latest version of the acclaimed real-time object detection and image segmentation model.

 ![example image](https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png)

 ### Sample output for detecting people (class = 0) in an image with 1 person
 ```
 {
    "boxes": [
        {
            "xyxy": array([[381.72784, 43.06653, 862.7265, 630.10913]], dtype=float32),
            "xywh": array([[622.2272, 336.58783, 480.99866, 587.0426]], dtype=float32),
            "xyxyn": array(
                [[0.2982249, 0.05981462, 0.6740051, 0.8751516]], dtype=float32
            ),
            "xywhn": array(
                [[0.48611498, 0.4674831, 0.3757802, 0.81533694]], dtype=float32
            ),
            "conf": array([0.93477935], dtype=float32),
            "cls": array([0.0], dtype=float32),
        }
    ]
}

 ```