import sieve
import cv2
import tempfile
import zipfile
import os



def is_video(file: sieve.File):
    file_path = file.path

    video_formats = ['mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv']

    if file_path.split(".")[-1] in video_formats:
        return True

    return False


def get_first_frame(video: sieve.File):
    video_path = video.path

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('first_frame.png', frame)
    else:
        raise Exception("Failed to read the video; empty or does not exist")

    frame = sieve.File(path='first_frame.png')
    cap.release()

    return frame


def zip_to_mp4(frames_zip: sieve.File):
    """
    convert zip file of frames to an mp4
    """
    output_path = "output_video.mp4"
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(frames_zip.path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split('_')[1]))

        first_frame = cv2.imread(os.path.join(temp_dir, images[0]))
        height, width, layers = first_frame.shape
        frame_size = (width, height)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)

        # Loop through the images and write them to the video
        for image in images:
            img_path = os.path.join(temp_dir, image)
            frame = cv2.imread(img_path)
            out.write(frame)

    out.release()
    return sieve.File(path=output_path)


def splice_audio(video, audio):
    spliced_path = "spliced.mp4"
    cmd = f"ffmpeg -y -nostdin -loglevel error -i {video.path} -i {audio.path} -c:v copy -c:a aac {spliced_path}"
    os.system(cmd)

    return sieve.File(path=spliced_path)


def resize_and_crop(image, target_width, target_height):
    """
    resize image to meet target_height, target_width without stretching
    """

    image_height, image_width = image.shape[:2]

    target_aspect = target_width / target_height
    image_aspect = image_width / image_height

    if image_aspect > target_aspect:
        new_height = target_height
        new_width = int(image_aspect * new_height)
    else:
        new_width = target_width
        new_height = int(new_width / image_aspect)

    resized_image = cv2.resize(image, (new_width, new_height))

    crop_x = (new_width - target_width) // 2
    crop_y = (new_height - target_height) // 2

    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image


def resize_with_padding(image, scale):
    h, w = image.shape[:2]
    
    # Calculate new dimensions based on the scale
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding to maintain original aspect ratio
    target_size = max(h, w)
    delta_h, delta_w = target_size - new_h, target_size - new_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    return padded
