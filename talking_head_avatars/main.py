import sieve
import imageio
import uuid

def compute_bbox(tube_bbox, frame_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])


    return (left, top, right, bot)

@sieve.Model(
    name="thin_plate_spline_motion_model",
    gpu = True,
    machine_type="a100",
    python_version="3.9",
    python_packages=[
        "requests==2.28.1",
        "imageio[ffmpeg]==2.22.4",
        "librosa==0.9.2",
        "numba==0.56.4",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "mediapipe==0.9.0.1",
        "av==9.2.0",
        "scikit-image==0.18.3",
        "matplotlib==3.4.3",
        "face-alignment==1.3.5",
        "ffmpeg-python==0.2.0"
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libavcodec58",
        "ffmpeg",
        "ninja-build"
    ],
    run_commands=[
        "mkdir -p /root/.cache/lip/models/",
        "wget -c 'https://storage.googleapis.com/sieve-public-model-assets/thinplate/vox.pth.tar' -P /root/.cache/lip/models/"
    ],
    iterator_input=True,
    persist_output=True
)


class ThinplateAvatar:
    def __setup__(self):
        import mediapipe as mp
        import cv2
        from thinplate import PlateTalkingHead
        self.model = PlateTalkingHead()
        self.model.setup('/root/.cache/lip/models/vox.pth.tar')
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    def detect_faces(self, img):
        import cv2
        import subprocess
        results = self.face_detection.process(img)
        img_height, img_width, _ = img.shape
        outputs = []
        if results.detections:
            for detection in results.detections:
                ratio_box = [detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height]
                # box x1, y1, x2, y2
                box = [int(ratio_box[0] * img_width), int(ratio_box[1] * img_height), int((ratio_box[0] + ratio_box[2]) * img_width), int((ratio_box[1] + ratio_box[3]) * img_height)]

                # increase box y1 by 20% and make sure it is not negative
                box[1] = max(0, int(box[1] - 0.2 * (box[3] - box[1])))

                # make width and height the larger of the two
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width > height:
                    box[1] = max(0, int(box[1] - 0.5 * (width - height)))
                    box[3] = min(img_height, int(box[3] + 0.5 * (width - height)))
                else:
                    box[0] = max(0, int(box[0] - 0.5 * (height - width)))
                    box[2] = min(img_width, int(box[2] + 0.5 * (height - width)))

                outputs.append({
                    "box": box,
                    "class_name": "face",
                    "score": detection.score[0],
                    "frame_number": None if not hasattr(img, "frame_number") else img.frame_number
                })
        return outputs
    def __predict__(self, imgs: sieve.Image, videos: sieve.Video) -> sieve.Video:
        import cv2
        import subprocess
        
        for img, video in zip(imgs, videos):
            print(img.width, img.height)
            print(video, video.path, video.fps, video.width, video.height)
            reader = imageio.get_reader(str(video.path), format='mp4')
            fps = reader.get_meta_data()["fps"]
            num_frames = reader.get_length()

            count = 0
            frames = []
            for im in reader:
                if count == 0:
                    # detect face in first frame
                    print("Detecting face in first frame")
                    faces = self.detect_faces(im)
                    if len(faces) == 0:
                        raise Exception("No faces detected")
                    
                    face = faces[0]
                    face_box = face['box']
                    face_box = compute_bbox(face_box, im.shape)

                    print("Cropping video to face")
                
                frame = im[face_box[1]:face_box[3], face_box[0]:face_box[2]]
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)

                count += 1

            reader.close()
            print("Running model")
            print(len(frames), fps)
            out_path = self.model.predict(img.path, frames, fps)

            print("Merging audio and video")
            final_path = f'{uuid.uuid4()}.mp4'
            command = 'ffmpeg -i \'{}\' -i \'{}\' -c copy -map 0:0 -map 1:1 -shortest \'{}\''.format(out_path, video.path, final_path)

            subprocess.call(command, shell=True)

            yield sieve.Video(path=final_path)

@sieve.workflow(name='thinplate_talking_head')
def thinplate_talking_head(img: sieve.Image, video: sieve.Video) -> sieve.Video:
    return ThinplateAvatar()(img, video)

@sieve.workflow(name='talking_head_avatar_generation')
def thinplate_talking_head(driving_video: sieve.Video, driving_audio: sieve.Audio, avatar_image: sieve.Image) -> sieve.Video:
    images = sieve.reference("sieve-developer/video-splitter")(driving_video)
    faces = sieve.reference("sieve-developer/mediapipe-face-detector")(images)
    tracked_faces = sieve.reference("sieve-developer/sort")(faces)
    synced = sieve.reference("sieve-developer/wav2lip")(driving_video, driving_audio, tracked_faces)
    return ThinplateAvatar()(avatar_image, synced)
