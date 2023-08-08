"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import cv2
import time
import numpy as np
from face_model.face_gan import FaceGAN
from align_faces import warp_and_crop_face, get_reference_facial_points
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection


class FaceEnhancement(object):
    def __init__(self):
        in_size = 512
        model = "GPEN-BFR-512"
        self.facegan = FaceGAN("./", in_size, in_size, model, 2, 1, None, device="cuda")
        self.mpface = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.01
        )
        self.in_size = in_size
        self.out_size = in_size
        self.facemesh = mpFaceMesh.FaceMesh(max_num_faces=1)
        self.threshold = 0.1
        self.alpha = 1

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 4)

        self.kernel = np.array(
            ([0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]),
            dtype="float32",
        )

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
            (self.in_size, self.in_size),
            inner_padding_factor,
            outer_padding,
            default_square,
        )

    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def process(self, img, facebs):
        orig_faces, enhanced_faces = [], []
        start_face = time.time()
        # if face == None:
        #     results = self.mpface.process(img)
        #     detection = results.detections[0]
        #     bb = detection.location_data.relative_bounding_box
        #     w = img.shape[1]
        #     h = img.shape[0]
        #     facebs = np.array([
        #         [bb.xmin * w, h * bb.ymin, w *(bb.xmin + bb.width), h * (bb.ymin + bb.height), detection.score[0]]
        #     ], dtype=np.float32)
        # facebs = face
        end_face = time.time()
        start_mesh = time.time()
        results = self.facemesh.process(img)
        landmarks = results.multi_face_landmarks[0].landmark
        landms = np.array(
            [
                [
                    [
                        img.shape[1] * (landmarks[33].x + landmarks[133].x) / 2,
                        img.shape[1] * (landmarks[362].x + landmarks[263].x) / 2,
                        img.shape[1] * landmarks[19].x / 1,
                        img.shape[1] * landmarks[61].x / 1,
                        img.shape[1] * landmarks[291].x / 1,
                    ],
                    [
                        img.shape[0] * (landmarks[159].y + landmarks[145].y) / 2,
                        img.shape[0] * (landmarks[386].y + landmarks[374].y) / 2,
                        img.shape[0] * landmarks[19].y / 1,
                        img.shape[0] * landmarks[61].y / 1,
                        img.shape[0] * landmarks[291].y / 1,
                    ],
                ]
            ],
            dtype=np.float32,
        )
        end_mesh = time.time()

        start_create = time.time()
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)
        end_create = time.time()

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < self.threshold:
                continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])

            start_reshape = time.time()
            facial5points = np.reshape(facial5points, (2, 5))
            end_reshape = time.time()

            start_warp = time.time()
            of, tfm_inv = warp_and_crop_face(
                img,
                facial5points,
                reference_pts=self.reference_5pts,
                crop_size=(self.in_size, self.in_size),
            )
            end_warp = time.time()
            # enhance the face
            start_ai = time.time()
            ef = self.facegan.process(of)
            end_ai = time.time()
            orig_faces.append(of)
            enhanced_faces.append(ef)

            start_mask = time.time()
            tmp_mask = self.mask
            # tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0]/255.)
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            end_mask = time.time()

            if min(fh, fw) < 100:  # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            # ef = cv2.addWeighted(ef, self.alpha, of, 1.-self.alpha, 0.0)

            start_mask1 = time.time()
            start_warp_affine_inv = time.time()
            if self.in_size != self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)
            end_warp_affine_inv = time.time()

            start_mask_sub = time.time()
            mask = tmp_mask - full_mask
            end_mask_sub = time.time()
            start_mask_a = time.time()
            mask_positive = np.where(mask > 0)
            full_mask[mask_positive] = tmp_mask[mask_positive]
            end_mask_a = time.time()
            start_mask_b = time.time()
            full_img[mask_positive] = tmp_img[mask_positive]
            end_mask_b = time.time()
            end_mask1 = time.time()

        start_end = time.time()
        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)
        end_end = time.time()

        # print('------------')
        # print('face', end_face - start_face)
        # print('face', end_mesh - start_mesh)
        # print('create', end_create - start_create)
        # print('reshape', end_reshape - start_reshape)
        # print('warp', end_warp - start_warp)
        # print('ai', end_ai - start_ai)
        # print('mask', end_mask - start_mask)
        # print('warp_aff', end_warp_affine_inv - start_warp_affine_inv)
        # print('mask_sub', end_mask_sub - start_mask_sub)
        # print('maska', end_mask_a - start_mask_a)
        # print('maskb', end_mask_b - start_mask_b)
        # print('mask1', end_mask1 - start_mask1)
        # print('end', end_end - start_end)
        # print('------------')
        return img, orig_faces, enhanced_faces
