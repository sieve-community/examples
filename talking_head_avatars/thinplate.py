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
from ffhq_dataset.face_alignment import image_align, image_align_rotate
from ffhq_dataset.landmarks_detector import LandmarksDetector

import cv2


warnings.filterwarnings("ignore")

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
        dataset_name = "vox"

        predict_mode = "relative"

        pixel = 384 if dataset_name == "ted" else 256

        if dataset_name == "vox":
            # first run face alignment
            angle, crop = align_image_rotate(str(source_image), 'aligned.png')

            # load source image and align it
            whole_image = cv2.imread(str(source_image))
            center = ((crop[3] - crop[1]) // 2, (crop[2] - crop[0]) // 2)
            h, w = whole_image.shape[:2]
            image_center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, (angle), 1.0)
            abs_cos = abs(M[0,0]) 
            abs_sin = abs(M[0,1])
            bound_w = int(h * abs_sin + w * abs_cos)
            bound_h = int(h * abs_cos + w * abs_sin)
            M[0, 2] += bound_w/2 - image_center[0]
            M[1, 2] += bound_h/2 - image_center[1]
            whole_image = cv2.warpAffine(whole_image, M, (bound_w, bound_h))
            reverse_angle_matrix = cv2.getRotationMatrix2D(center, (-angle), 1.0)
            whole_image = cv2.cvtColor(whole_image, cv2.COLOR_BGR2RGB)
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

        border_crop = 0.1
        for i in range(len(predictions)):
            talking = img_as_ubyte(cv2.resize(predictions[i], (crop[2] - crop[0], crop[3] - crop[1])))
            tmp = whole_image.copy()
            tmp[crop[1]:crop[3], crop[0]:crop[2], :] = talking
            tmp = cv2.warpAffine(tmp, reverse_angle_matrix, (tmp.shape[1], tmp.shape[0]))

            tmp = tmp[int(tmp.shape[0] * border_crop):int(tmp.shape[0] * (1 - border_crop)), int(tmp.shape[1] * border_crop):int(tmp.shape[1] * (1 - border_crop)), :]

            predictions[i] = img_as_ubyte(tmp)

        out_path = f"{uuid.uuid4()}.mp4"

        imageio.mimsave(
            str(out_path), [frame for frame in predictions], fps=fps
        )
        return out_path


def align_image_rotate(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        return image_align_rotate(raw_img_path, aligned_face_path, face_landmarks)
