import cv2
import numpy as np

######### face enhancement
from face_parse.face_parsing import FaceParse
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
# from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# from numba import jit
facemesh = mpFaceMesh.FaceMesh(max_num_faces=1)
def get_bounding_face_mask(image, size=512):
    image = cv2.resize(image, (size, size))
    results = facemesh.process(image)
    lms = results.multi_face_landmarks
    
    if lms == None:
        # Create an empty mask with same dimensions as the image
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        return mask
    else:
        lms = lms[0].landmark
    mouth = [
        234,
        93,
        132,
        58,
        172,
        136,
        150,
        149,
        176,
        148,
        152,
        377,
        400,
        378,
        379,
        365,
        397,
        288,
        361,
        323,
        454,
        356,
        389,
        251,
        284,
        332,
        297,
        338,
        10,
        109,
        67,
        103,
        54,
        21,
        162,
        127
    ]
    lms1 = np.array([[int(512 * (lms[x].x)), int(512 * (lms[x].y))] for x in mouth])
    
    # scale points to 512

    def create_mask(image, points):
        # Convert points to numpy array with required format
        points = np.array(points, dtype=np.int32)

        # Create an empty mask with same dimensions as the image
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Find the bounding polygon
        hull = cv2.convexHull(points)

        # Draw the filled polygon on the mask
        cv2.fillConvexPoly(mask, hull, (255))
    
        return mask

    out = create_mask(image, lms1)
    # print(out.shape)
    return out

class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, use_sr=True, sr_model=None, channel_multiplier=2, narrow=1, device='cuda'):
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier, narrow, device=device)
        # self.srmodel =  RealESRNet(base_dir, sr_model, device=device)
        self.srmodel=None
        self.faceparser = FaceParse(base_dir, device=device)
        self.use_sr = use_sr
        self.size = size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=20):
        # Vectorized thresholding
        mask[:thres,:] = 0
        mask[-thres:,:] = 0 
        mask[:,:thres] = 0
        mask[:,-thres:] = 0

        # Faster Gaussian blur 
        mask = cv2.GaussianBlur(mask, (51, 51), 5)

        return mask.astype(np.float32)
    
    def process(self, img, ori_img, bbox=None, faceb=None, landm=None, face_enhance=True, possion_blending=False, cox=None):
        import time
        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])

        # facebs = np.array([[bbox[2], bbox[0], bbox[3], bbox[1], 1.0]])

        # ox1, ox2, oy1, oy2 = cox

        # right_eye_points = [43, 47, 44, 46]
        # left_eye_points = [38, 40, 37, 41]
        # right_eye_center = np.mean(landm[left_eye_points], axis=0)
        # left_eye_center = np.mean(landm[right_eye_points], axis=0)
        # nose_center = landm[30]
        # right_lip_center = landm[48]
        # left_lip_center = landm[54]

        # landms = [[
        #     0, 0,
        #     0, 0,
        #     0, 0,
        #     0, 0,
        #     0, 0
        # ]]

        # for mark, ind in zip([left_eye_center, right_eye_center, nose_center, left_lip_center, right_lip_center], [(0,5), (1,6), (2,7), (3,8), (4,9)]):
        #     x = mark[0]
        #     y = mark[1]
        #     box_h = ox2 - ox1
        #     box_w = oy2 - oy1
        #     # above coords are from 256x256, need to scale back to cox
        #     x = x * box_h / 256
        #     y = y * box_w / 256

        #     # the tranform above is upside down, need to flip
        #     # x = box_h - x
        #     y = box_w - y
            

        #     # add bbox offset
        #     x += ox1
        #     y += oy1

        #     landms[0][ind[0]] = x
        #     landms[0][ind[1]] = y

        # facebs, landms = self.facedetector.detect(img.copy())
        # print('-'*80)
        t = time.time()
        if faceb is not None and landm is not None:
            facebs, landms = faceb, landm
        else:
            facebs, landms = self.facedetector.detect(img.copy())
        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(ori_img.shape, dtype=np.uint8)

        if len(facebs)==0:
            mask_sharp = np.zeros((height, width), dtype=np.float32)
        
        # print('zero', time.time() - t)
        t = time.time()

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            t = time.time()
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            # print('one', time.time() - t)
            t = time.time()

            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))

            # print('warp crop', time.time() - t)
            t = time.time()

            # enhance the face
            if face_enhance:
                ef = self.facegan.process(of)
            else:
                ef = of
                    
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            # print('enhance', time.time() - t)
            t = time.time()
            # print(ef.shape)
            # tmp_mask = self.mask
            '''
            0: 'background' 1: 'skin'   2: 'nose'
            3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
            6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
            9: 'r_ear'  10: 'mouth' 11: 'u_lip'
            12: 'l_lip' 13: 'hair'  14: 'hat'
            15: 'ear_r' 16: 'neck_l'    17: 'neck'
            18: 'cloth'
            '''

            # no ear, no neck, no hair&hat,  only face region
            mm = [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
            # mask_sharp = self.faceparser.process(ef, mm)[0]/255.
            mask_sharp = get_bounding_face_mask(ef)/255.
            # print('faceparse', time.time() - t)
            t = time.time()
            tmp_mask = self.mask_postprocess(mask_sharp)
            # print('mask post', time.time() - t)
            t = time.time()
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            mask_sharp = cv2.resize(mask_sharp, ef.shape[:2])
            # print('resize', time.time() - t)
            t = time.time()

            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            mask_sharp = cv2.warpAffine(mask_sharp, tfm_inv, (width, height), flags=3)
            # print('warp aff', time.time() - t)
            t = time.time()

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            if face_enhance:
                tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)
            else:
                tmp_img = cv2.warpAffine(of, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            tmp_where = np.where(mask>0)
            full_mask[tmp_where] = tmp_mask[tmp_where]
            full_img[tmp_where] = tmp_img[tmp_where]

            # print('things', time.time() - t)
            t = time.time()

        mask_sharp = cv2.GaussianBlur(mask_sharp, (0,0), sigmaX=1, sigmaY=1, borderType = cv2.BORDER_DEFAULT)

        # print('gauss', time.time() - t)
        t = time.time()

        full_mask = full_mask[:, :, np.newaxis]
        mask_sharp = mask_sharp[:, :, np.newaxis]

        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)

        elif possion_blending is True:
            if bbox is not None:
                y1, y2, x1, x2 = bbox
                mask_bbox = np.zeros_like(mask_sharp)
                mask_bbox[y1:y2 - 5, x1:x2] = 1
                full_img, ori_img, full_mask = [cv2.resize(x,(512,512)) for x in (full_img, ori_img, np.float32(mask_sharp * mask_bbox))]
                # print('rsize', time.time() - t)
                t = time.time()
            else:
                full_img, ori_img, full_mask = [cv2.resize(x,(512,512)) for x in (full_img, ori_img, full_mask)]
                # print('rsize1', time.time() - t)
                t = time.time()
            
            img = Laplacian_Pyramid_Blending_with_mask(full_img, ori_img, full_mask, 6)
            img = np.clip(img, 0 ,255)
            img = np.uint8(cv2.resize(img, (width, height)))
            # print('laprsize', time.time() - t)
            t = time.time()

        else:
            img = cv2.convertScaleAbs(ori_img*(1-full_mask) + full_img*full_mask)
            img = cv2.convertScaleAbs(ori_img*(1-mask_sharp) + img*mask_sharp)

            # print('convert', time.time() - t)

        # print('--' * 40)
        return img, orig_faces, enhanced_faces