import sieve
from typing import Dict

model_metadata = sieve.Metadata(
    description="Track a point through a video using feature matching.",
    code_url="https://github.com/sieve-community/examples/blob/main/point_tracking/main.py",
    tags=["Tracking", "Video"],
    readme=open("README.md", "r").read(),
)


@sieve.Model(
    name="klt_superglue_point_tracking",
    iterator_input=True,
    python_packages=["torch==1.13.1", "torchvision==0.14.1"],
    run_commands=[
        "mkdir -p /root/.cache/superglue/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/superglue/superglue_outdoor.pth' -P /root/.cache/superglue/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/superglue/superglue_indoor.pth' -P /root/.cache/superglue/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/superglue/superpoint_v1.pth' -P /root/.cache/superglue/models",
    ],
    metadata=model_metadata,
)
class PointTrackingSuperGlue:
    def __setup__(self):
        import torch
        from models.matching import Matching

        torch.set_grad_enabled(False)

        self.resize = [640, 480]
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = -1
        force_cpu = True
        import torch

        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print('Running inference on device "{}"'.format(self.device))
        config = {
            "superpoint": {
                "nms_radius": nms_radius,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.2,
            },
        }
        self.matching = Matching(config).eval().to(self.device)
        self.keys = ["keypoints", "scores", "descriptors"]

    def __predict__(self, video: sieve.Video, xs: int, ys: int) -> Dict:
        """
        :param video: A video to track points in
        :param xs: The x coordinate to the point to track
        :param ys: The y coordinate to the point to track
        :return: A dictionary with the x and y coordinates of the tracked point in each frame
        """

        interval = 60
        from models.utils import frame2tensor
        import cv2
        import numpy as np
        import torch

        def transform_point(x, y, pts0, pts1):
            # Calculate the homography
            H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)

            # Prepare the target point to be transformed
            target_point = np.array([[x, y]], dtype=np.float32)
            target_point = np.column_stack(
                (target_point, np.ones(target_point.shape[0]))
            )

            # Apply the homography to the target point
            transformed_point = H @ target_point.T
            transformed_point = transformed_point / transformed_point[2]

            # Get x and y coordinates of the transformed point
            x_transformed, y_transformed = (
                transformed_point[0][0],
                transformed_point[1][0],
            )

            return x_transformed, y_transformed

        def add_xy_point_to_keypoints(last_data, x_resized, y_resized):
            print(last_data)
            new_keypoint = torch.tensor(
                [x_resized, y_resized], dtype=torch.float32
            ).unsqueeze(0)

            # Access the tensor directly from the list and concatenate with the new_keypoint
            last_data["keypoints0"] = (
                torch.cat((last_data["keypoints0"][0], new_keypoint), dim=0),
            )  # Encapsulate result in a tuple

            # Update scores0 tensor
            new_score = torch.tensor([1.0], dtype=torch.float32)
            last_data["scores0"] = (
                torch.cat((last_data["scores0"][0], new_score), dim=0),
            )  # Encapsulate result in a tuple

            # Update descriptors0 tensor
            new_descriptor = torch.zeros((256, 1), dtype=torch.float32)
            last_data["descriptors0"] = (
                torch.cat((last_data["descriptors0"][0], new_descriptor), dim=1),
            )  # Encapsulate result in a tuple

            return last_data

        def use_xy_point(last_data, x_resized, y_resized):
            print(last_data)
            new_keypoint = torch.tensor(
                [[x_resized, y_resized]], dtype=torch.float32
            ).unsqueeze(0)

            # Access the tensor directly from the list and concatenate with the new_keypoint
            last_data["keypoints0"] = (new_keypoint,)  # Encapsulate result in a tuple

            # Update scores0 tensor
            new_score = torch.tensor([[1.0]], dtype=torch.float32)
            last_data["scores0"] = (new_score,)  # Encapsulate result in a tuple

            # Update descriptors0 tensor
            new_descriptor = torch.zeros((1, 1, 256), dtype=torch.float32)
            last_data["descriptors0"] = (
                new_descriptor,
            )  # Encapsulate result in a tuple

            return last_data

        def get_corresponding_keypoint(matchidx, kpts1):
            if matchidx > -1:
                return kpts1[matchidx]
            return None

        def get_corresponding_score(matchidx, scores1):
            if matchidx > -1:
                return scores1[matchidx]
            return None

        for vid, x, y in zip(video, xs, ys):
            cap = cv2.VideoCapture(vid.path)

            ret, frame = cap.read()

            resized_frame = cv2.resize(frame.copy(), (self.resize[0], self.resize[1]))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            frame_tensor = frame2tensor(resized_frame, self.device)
            last_data = self.matching.superpoint({"image": frame_tensor})
            last_data = {k + "0": last_data[k] for k in self.keys}
            last_data["image0"] = frame_tensor
            print(last_data)

            x_resized, y_resized = int(x * self.resize[0] / vid.width), int(
                y * self.resize[1] / vid.height
            )
            last_data = add_xy_point_to_keypoints(last_data, x_resized, y_resized)
            # last_data['keypoints0'] = [torch.tensor([[x_resized, y_resized]], dtype=torch.float32)]
            # last_data['scores0'] = (torch.tensor([[1.0]], dtype=torch.float32))
            # last_data['descriptors0'] = (torch.zeros((1, 1, 256), dtype=torch.float32))

            curr_score = 0

            last_frame = resized_frame
            last_image_id = 0

            point = [np.array([[x, y]], dtype=np.float32)]
            # Define LK parameters
            lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            )

            old_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

            count = 0
            while True:
                print(count)
                ret, frame = cap.read()

                if not ret:
                    break

                # frame = cv2.resize(frame.copy(), (self.resize[0], self.resize[1]))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate the optical flow
                new_point, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, point[0], None, **lk_params
                )

                if status[0]:
                    point[0] = new_point
                    x_temp, y_temp = point[0].ravel()
                    x_temp, y_temp = int(x_temp), int(y_temp)
                    # x_resized = x# * (frame.shape[1] / self.resize[0])
                    # y_resized = y# * (frame.shape[0] / self.resize[1])
                    # x, y = int(x_resized), int(y_resized)
                    yield {
                        "x": x_temp,
                        "y": y_temp,
                        "frame": count,
                        "status": "Tracked point",
                    }
                else:
                    if not x and not y:
                        yield {
                            "status": "Unable to track point",
                            "x": -1,
                            "y": -1,
                            "frame": count,
                        }
                    else:
                        x_temp, y_temp = point[0].ravel()
                        x_temp, y_temp = int(x_temp), int(y_temp)
                        yield {
                            "status": "Tracked point",
                            "x": x_temp,
                            "y": y_temp,
                            "frame": count,
                        }

                if count % interval == 0:
                    proposed_point_1 = None
                    proposed_point_1_score = 0
                    proposed_point_2 = None
                    proposed_point_2_score = 0

                    x_resized = x * (self.resize[0] / frame.shape[1])
                    y_resized = y * (self.resize[1] / frame.shape[0])

                    # Add the x, y point to the last_data keypoints
                    last_data_updated = last_data

                    resized_frame_gray = cv2.resize(
                        old_gray.copy(), (self.resize[0], self.resize[1])
                    )
                    frame_tensor = frame2tensor(resized_frame_gray, self.device)
                    # pred = self.matching({**last_data_updated, 'image1': frame_tensor})

                    pred = self.matching({**last_data, "image1": frame_tensor})

                    kpts0 = last_data_updated["keypoints0"][0].detach().cpu().numpy()
                    kpts1 = pred["keypoints1"][0].detach().cpu().numpy()
                    matches = pred["matches0"][0].detach().cpu().numpy()
                    scores = pred["scores1"][0].detach().cpu().numpy()
                    descriptors0 = pred["descriptors1"][0].detach().cpu().numpy()
                    # print('scores', scores)
                    # # Find the corresponding match for the added keypoint
                    added_keypoint_match_idx = matches[-1]
                    added_keypoint_corresponding_kpt = get_corresponding_keypoint(
                        added_keypoint_match_idx, kpts1
                    )
                    added_keypoint_corresponding_score = get_corresponding_score(
                        added_keypoint_match_idx, scores
                    )
                    if added_keypoint_corresponding_kpt is not None:
                        # If a match is found, convert the coordinates back to original image size
                        x_transformed, y_transformed = added_keypoint_corresponding_kpt

                        # last_data['keypoints0'] = pred['keypoints1']
                        # last_data['scores0'] = pred['scores1']
                        # last_data['descriptors0'] = pred['descriptors1']

                        x_transformed = x_transformed * (
                            frame.shape[1] / self.resize[0]
                        )
                        y_transformed = y_transformed * (
                            frame.shape[0] / self.resize[1]
                        )

                        proposed_point_1 = np.array(
                            [[x_transformed, y_transformed]], dtype=np.float32
                        )
                        proposed_point_1_score = added_keypoint_corresponding_score
                    # else:

                    # kpts0 = last_data['keypoints0'][0].cpu().numpy()
                    # kpts1 = pred['keypoints1'][0].cpu().numpy()
                    # matches = pred['matches0'][0].cpu().numpy()
                    # Create two ordered lists of points based on the matches
                    pts0, pts1 = [], []
                    for m in range(len(matches)):
                        if matches[m] > -1:
                            pts0.append(kpts0[m])
                            pts1.append(kpts1[matches[m]])

                    pts0, pts1 = np.array(pts0), np.array(pts1)

                    x_resized = x * (self.resize[0] / frame.shape[1])
                    y_resized = y * (self.resize[1] / frame.shape[0])

                    if len(pts0) >= 4 and len(pts1) >= 4:
                        x_transformed, y_transformed = transform_point(
                            x_resized, y_resized, pts0, pts1
                        )
                        x_transformed = x_transformed * (
                            frame.shape[1] / self.resize[0]
                        )
                        y_transformed = y_transformed * (
                            frame.shape[0] / self.resize[1]
                        )

                        proposed_point_2 = np.array(
                            [[x_transformed, y_transformed]], dtype=np.float32
                        )
                        # average all the scores
                        proposed_point_2_score = np.mean(scores)

                    if proposed_point_1 is not None and proposed_point_2 is not None:
                        # pick the point that is closest to the curr point
                        curr_point = point[0]
                        proposed_point = None
                        proposed_score = 0
                        if np.linalg.norm(
                            proposed_point_1 - curr_point
                        ) < np.linalg.norm(proposed_point_2 - curr_point):
                            proposed_point = proposed_point_1
                            proposed_score = proposed_point_1_score
                        else:
                            proposed_point = proposed_point_2
                            proposed_score = proposed_point_2_score
                    elif proposed_point_1 is not None:
                        proposed_point = proposed_point_1
                        proposed_score = proposed_point_1_score
                    elif proposed_point_2 is not None:
                        proposed_point = proposed_point_2
                        proposed_score = proposed_point_2_score

                    if proposed_point is not None:
                        # thresh = 0.05 * (frame.shape[1] + frame.shape[0]) / 2
                        # if np.linalg.norm(proposed_point - point[0]) < thresh:
                        point[0] = proposed_point
                        # if proposed_score > curr_score:
                        #     last_data['keypoints0'] = pred['keypoints1']
                        #     last_data['scores0'] = pred['scores1']
                        #     last_data['descriptors0'] = pred['descriptors1']

                old_gray = frame_gray.copy()

                count += 1

            cap.release()


metadata = sieve.Metadata(
    title="Motion Tracking",
    description="Track a point through a video.",
    code_url="https://github.com/sieve-community/examples/blob/main/point_tracking/main.py",
    tags=["Tracking", "Video"],
    readme=open("README.md", "r").read(),
)


@sieve.workflow(name="point-tracking", metadata=metadata)
def point_tracking(video: sieve.Video, x: int, y: int) -> Dict:
    return PointTrackingSuperGlue()(video, x, y)


if __name__ == "__main__":
    sieve.push(
        workflow="point-tracking",
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/bike.mp4"
            },
            "x": 100,
            "y": 100,
        },
    )
