import mediapipe as mp
import cv2
import itertools
import numpy as np
mp_face_detection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh
class LandmarksDetector:
    def __init__(self):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.facemesh = mpFaceMesh.FaceMesh(max_num_faces=1)
        # self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used

    def get_landmarks(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.asarray(image)
            # convert imageio to cv2
        # dets = self.detector.process(img)
        # dtss = []
        # for detection in dets.detections:
        #     box = detection.location_data.relative_bounding_box
        #     x1, y1, w, h = box.xmin, box.ymin, box.width, box.height
        #     x2, y2 = x1 + w, y1 + h
        #     x1 = x1 * img.shape[1]
        #     x2 = x2 * img.shape[1]
        #     y1 = y1 * img.shape[0]
        #     y2 = y2 * img.shape[0]
        #     dtss.append(dlib.rectangle(int(x1), int(y1), int(x2), int(y2)))
        # results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print(dets)
        results = self.facemesh.process(img)
        faces = results.multi_face_landmarks
        for lm in faces:
            landmarks = lm.landmark
            ## just get face parts
            # tmp = [landmarks[0:17], landmarks[17:22], landmarks[22:27], landmarks[27:31], landmarks[31:36], landmarks[36:42], landmarks[42:48], landmarks[48:60], landmarks[60:68]]
            # landmarks = []
            # for i in range(len(tmp)):
            #     for j in range(len(tmp[i])):
            #         landmarks.append(tmp[i][j])
            # idxs = [33, 263, 61, 291, 199]
            # landmarks = [landmarks[i] for i in idxs]
            # just get contours
            # face connections
            tmp = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
                # print(connection)
                # print(landmarks[connection[0]])
                # print(landmarks[connection[1]])
                # cv2.line(img, (int(img.shape[1] * landmarks[connection[0]].x), int(img.shape[0] * landmarks[connection[0]].y)), (int(img.shape[1] * landmarks[connection[1]].x), int(img.shape[0] * landmarks[connection[1]].y)), (0, 255, 0), 1)
            lnds = []
            for i in tmp:
                lnds.append(landmarks[i])
            landmarks = lnds
            # print(landmarks)
            face_landmarks = []
            for landmark in landmarks:
                # print(img.shape[1] * landmark.x, img.shape[0] * landmark.y)
                face_landmarks.append((img.shape[1] * landmark.x, img.shape[0] * landmark.y))
                cv2.circle(img, (int(img.shape[1] * landmark.x), int(img.shape[0] * landmark.y)), 1, (0, 0, 255), -1)
            cv2.imwrite('test.png', img)
            # landms = np.array([[
            #     [
            #         img.shape[1] * (landmarks[33].x + landmarks[133].x) / 2,
            #         img.shape[1] * (landmarks[362].x + landmarks[263].x) / 2,
            #         img.shape[1] * landmarks[19].x / 1,
            #         img.shape[1] * landmarks[61].x / 1,
            #         img.shape[1] * landmarks[291].x / 1,
            #     ],
            #     [
            #         img.shape[0] * (landmarks[159].y + landmarks[145].y) / 2,
            #         img.shape[0] * (landmarks[386].y + landmarks[374].y) / 2,
            #         img.shape[0] * landmarks[19].y / 1,
            #         img.shape[0] * landmarks[61].y / 1,
            #         img.shape[0] * landmarks[291].y / 1,
            #     ]
            # ]], dtype=np.float32)
            # face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
