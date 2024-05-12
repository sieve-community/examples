import cv2
from PIL import Image
from io import BytesIO
import tempfile
import numpy as np

def opencv_extract_frames(vpath_or_bytesio, sampling_count=6, fps=None, frame_count=None, sampling_strategy="auto"):
    """
    Extract frames from a video using OpenCV.

    Args:
        vpath_or_bytesio (str or BytesIO): Path to the video file or BytesIO object containing the video.
        frames (int): Number of frames to extract from the video.
        fps (int): Frames per second of the video. If None, it will be inferred from the video.
        frame_count (int): Number of frames in the video. If None, it will be inferred from the video.


    Returns:
        list: List of PIL Images extracted from the video.

    Raises:
        NotImplementedError: If the type of `vpath_or_bytesio` is not supported.
    """
    import cv2

    if isinstance(vpath_or_bytesio, str):
        vidcap = cv2.VideoCapture(vpath_or_bytesio)
        return get_frame_from_vcap(vidcap, sampling_count, fps=fps, frame_count=frame_count, sampling_strategy=sampling_strategy)
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # assuming mp4
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            return get_frame_from_vcap(vidcap, sampling_count, fps=fps, frame_count=frame_count, sampling_strategy=sampling_strategy)
    else:
        raise NotImplementedError(type(vpath_or_bytesio))


def get_frame_from_vcap(vidcap, num_frames=10, fps=None, frame_count=None, sampling_strategy="auto"):
    if fps is None or frame_count is None:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = vidcap.read()
        if success:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            images.append(im_pil)
        else:
            print(f"Skipping frame at index {frame_index} due to read failure.")

    if len(images) < num_frames:
        print("Did not find enough valid frames, returning what was collected.")

    return images

