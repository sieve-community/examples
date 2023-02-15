# sys.path.insert(0, "stylegan-encoder")
import warnings
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import uuid
import numpy as np

from demo import load_checkpoints
from demo import make_animation_batched
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


warnings.filterwarnings("ignore")

print('hey')

print('bye')

LANDMARKS_DETECTOR = LandmarksDetector()


class PlateTalkingHead():
    def setup(self, checkpoint_path):
        self.device = torch.device("cuda:0")
        datasets = ["vox"]
        (
            self.inpainting,
            self.kp_detector,
            self.dense_motion_network,
            self.avd_network,
        ) = ({}, {}, {}, {})
        for d in datasets:
            (
                self.inpainting[d],
                self.kp_detector[d],
                self.dense_motion_network[d],
                self.avd_network[d],
            ) = load_checkpoints(
                config_path=f"config/{d}-384.yaml"
                if d == "ted"
                else f"config/{d}-256.yaml",
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
        from face_enhancer import FaceEnhancement
        self.gpen = FaceEnhancement()

    def predict(self, source_image, driving_video, fps):
        orig_img = imageio.imread(str(source_image))
        dataset_name = "vox"

        predict_mode = "relative"  # ['standard', 'relative', 'avd']
        # find_best_frame = False

        pixel = 384 if dataset_name == "ted" else 256

        if dataset_name == "vox":
            # first run face alignment
            align_image(str(source_image), 'aligned.png')
            source_image = imageio.imread('aligned.png')
        else:
            source_image = imageio.imread(str(source_image))
        source_image = resize(source_image, (pixel, pixel))[..., :3]

        driving_video = [
            resize(frame, (pixel, pixel))[..., :3] for frame in driving_video
        ]

        inpainting, kp_detector, dense_motion_network, avd_network = (
            self.inpainting[dataset_name],
            self.kp_detector[dataset_name],
            self.dense_motion_network[dataset_name],
            self.avd_network[dataset_name],
        )

        predictions = make_animation_batched(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            device="cuda:0",
            mode=predict_mode,
        )
        
        temp_path = f'{uuid.uuid4()}.mp4'

        if self.gpen:
            for i, frame in enumerate(predictions):
                # make frame np uint8
                frame = img_as_ubyte(frame)
                img_out, _, _ = self.gpen.process(frame, np.array([
                    [0, 0, 256, 256, 0.99]
                ], dtype=np.float32))
                predictions[i] = img_out
                print(f'gpen {i}')
        
        # save resulting video
        out_path = f"{uuid.uuid4()}.mp4"

        imageio.mimsave(
            str(out_path), [frame for frame in predictions], fps=fps
        )
        return out_path


def align_image(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        image_align(raw_img_path, aligned_face_path, face_landmarks)

