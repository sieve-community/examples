#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import sieve

@sieve.Model(
    name="face-parsing-head-segmentation",
    python_packages=[
        "torch",
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "torchvision",
    ],
    system_packages=[
        "ffmpeg",
        "libx264-dev",
        "zip",
    ],
    python_version="3.10",
    cuda_version="11.8",
    gpu=sieve.gpu.L4(),
)
class HeadSegmentationModel:

    def __setup__(self):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        checkpoint_path = '79999_iter.pth'
        net.load_state_dict(torch.load(checkpoint_path))
        net.eval()

        self.net = net
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __predict__(self, file: sieve.File, debug_viz: bool = False) -> sieve.File:
        file_path = file.path

        is_video = False

        import os
        _, file_extension = os.path.splitext(file_path)
        if file_extension in ['.jpg', '.jpeg', '.png']:
            print("This is an image file.")
        elif file_extension in ['.mp4', '.avi', '.mov']:
            print("This is a video file.")
            is_video = True
        else:
            raise ValueError("Unsupported file format, must be one of: jpg, jpeg, png, mp4, avi, mov")
        
        import cv2

        if is_video:
            # Process video
            import numpy as np
            import cv2

            import time
            st = time.time()

            video = cv2.VideoCapture(file_path)
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            fps = video.get(cv2.CAP_PROP_FPS)
            size = (frame_width, frame_height)

            if debug_viz:
                if os.path.exists("temp_viz.mp4"):
                    os.remove("temp_viz.mp4")
                out_viz = cv2.VideoWriter('temp_viz.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            masks_dir = "masks"

            import shutil
            shutil.rmtree(masks_dir, ignore_errors=True)
            os.makedirs(masks_dir)

            counter = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break      
                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                vis_parsing_anno = self.get_parsing_anno(frame_image)
                if debug_viz:
                    vis_im = vis_parsing_maps(frame_image, vis_parsing_anno)
                    out_viz.write(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))

                cv2.imwrite(f"{masks_dir}/%06d.png" % counter, vis_parsing_anno)
                counter +=1 

            import subprocess

            video.release()

            if os.path.exists('masks.zip'):
                os.remove('masks.zip')

            command = "zip -r masks.zip masks"
            process = subprocess.Popen(command, shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            print("time to process: ", time.time() - st)

            if debug_viz:
                out_viz.release()
                command = "ffmpeg -loglevel error -y -i temp_viz.mp4 -c:v libx264 -crf 17 segmentation_map_viz.mp4"
                subprocess.call(command, shell=True)

                return (sieve.File(path="masks.zip"), sieve.File(path="segmentation_map_viz.mp4"))

            return sieve.File(path="masks.zip")
        else:
            image = Image.open(file.path)
            save_path = "save_path.jpg"
            save_path_viz = "save_path_viz.jpg"
            if os.path.exists(save_path):
                os.remove(save_path)
            if os.path.exists(save_path_viz):
                os.remove(save_path_viz)

            vis_parsing_anno = self.get_parsing_anno(image)
            cv2.imwrite(save_path, vis_parsing_anno)

            if debug_viz:
                vis_im = vis_parsing_maps(image, vis_parsing_anno)
                cv2.imwrite(save_path_viz, cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))

                return (sieve.File(path=save_path), sieve.File(path=save_path_viz))
            
            return sieve.File(path=save_path)
        
    def get_parsing_anno(self, image):
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        with torch.no_grad():
            out = self.net(img)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        return parsing.astype(np.uint8)


def vis_parsing_maps(im, vis_parsing_anno):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_im


if __name__ == "__main__":
    a = HeadSegmentationModel()
    a.__predict__(sieve.File(path="/home/abhinav_ayalur_gmail_com/examples/head_segmentation/face_parsing/hdtr.mp4"), debug_viz=False)
