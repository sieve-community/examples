import cv2
from PIL import Image
import numpy as np


def get_frame_from_vcap(vidcap, num_frames=10, sampling_strategy="auto"):
    fps = vidcap.get_avg_fps()
    frame_count = len(vidcap)
    
    if fps == 0 or frame_count == 0:
        print("Video file not found. Return empty images.")
        return [Image.new("RGB", (720, 720))] * num_frames
    
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("Frame_interval is equal to 0. Return empty image.")
        return [Image.new("RGB", (720, 720))] * num_frames
    
    images = []
    frame_indices = np.linspace(5, frame_count - 2, num_frames, dtype=int) if sampling_strategy == "auto" \
                    else [int(i) for i in sampling_strategy.split(",")]
    num_frames = len(frame_indices)

    for frame_index in frame_indices:
        frame = vidcap[frame_index].asnumpy()
        if frame is not None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            images.append(im_pil)
        else:
            print(f"Skipping frame at index {frame_index} due to read failure.")

    if len(images) < num_frames:
        print("Did not find enough valid frames, returning what was collected.")

    return images