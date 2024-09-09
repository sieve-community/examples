import sieve
import shutil
import os
import cv2

from sam_fx import run_all
from utils import get_first_frame

import config


def generate_thumbnails():
    config.CACHE = True

    os.makedirs("thumbnails", exist_ok=True)
    video = sieve.File(path="duckling.mp4")


    if not [x for x in os.listdir("outputs") if x.endswith(".mp4")]:
        run_all(video.path, "duckling")

    for output in [x for x in os.listdir("outputs") if x.endswith(".mp4")]:
        name = output.split(".")[0]
        first_frame = get_first_frame(sieve.File(path=os.path.join("outputs", output)))
        out_path = os.path.join("thumbnails", f"{name}.png")
        print(f"created {out_path}")
        shutil.move(first_frame.path, out_path)


    # crop all images to be a square (top half of image, using width as height)
    for thumbnail in [x for x in os.listdir("thumbnails") if x.endswith(".png")]:
        img = cv2.imread(os.path.join("thumbnails", thumbnail))
        h, w, _ = img.shape
        img = img[:w, :]
        cv2.imwrite(os.path.join("thumbnails", thumbnail), img)

    # further crop border on all sides by 10%
    for thumbnail in [x for x in os.listdir("thumbnails") if x.endswith(".png")]:
        img = cv2.imread(os.path.join("thumbnails", thumbnail))
        h, w, _ = img.shape
        img = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        cv2.imwrite(os.path.join("thumbnails", thumbnail), img)


if __name__ == "__main__":
    generate_thumbnails()


