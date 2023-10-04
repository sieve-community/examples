#TODO: do stuff

import sieve
import concurrent.futures

metadata = sieve.Metadata(
    title="Video Background Remover",
    description="Remove background from video",
    code_url="https://github.com/sieve-community/examples/tree/main/background_removal",
    image=sieve.Image(
        url="https://play-lh.googleusercontent.com/teK3vaRZmw_edxydd18as4gM2mkkw-43vH_45nyjLFJHnQVcfQktesWoTrPiFKRyKg"
    ),
    tags=["Video", "Background", "Removal", "Featured"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="video_background_remover",
    gpu = True,
    python_packages=[
        "six==1.16.0",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/dis/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/isnet-general-use.pth' -P /root/.cache/dis/models/"
    ],
    metadata=metadata
)
class VideoBackgroundRemover:
    def __setup__(self):
        from dis_model import Dis
        self.dis = Dis()

    def __predict__(self, video: sieve.Video) -> sieve.Video:
        video_path = video.path

        DIS_background_remover = self.dis

        # use ffmpeg to extract frames from video
        import subprocess
        import os
        import uuid

        temp_dir ='temp'
        # delete temp dir if it exists
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # extract frames from video
        subprocess.run(["ffmpeg", "-i", video_path, "-qscale:v", "2", f"{temp_dir}/%06d.jpg"])

        # get list of frames
        import glob
        frames = glob.glob(f"{temp_dir}/*.jpg")
        frames.sort()

        # remove background from each frame
        # remove background from each frame
        imgs = []
        for frame_num, frame in enumerate(frames):
            img = sieve.Image(path=frame)
            imgs.append(img)
            # print(f"Frame {frame_num} created for background remover")
        
        # make directory for output
        output_dir = 'output'
        # delete output dir if it exists
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # start joining frames back together as they finish
        import cv2
        writer = None
        import time
        with concurrent.futures.ThreadPoolExecutor() as executor:
            count = 0
            for job in executor.map(DIS_background_remover, imgs):
                t = time.time()
                for res in job:
                    break
                if writer is None:
                    height, width, layers = res.array.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(f'{output_dir}/output.mp4', fourcc, 30, (width,height))
                writer.write(res.array)
                print(f"Waited {time.time() - t} seconds for result {count}")
                count += 1
        writer.release()

        # delete temp dir
        import shutil
        shutil.rmtree(temp_dir)

        # run through ffmpeg to make sure the video is playable
        subprocess.run(["ffmpeg", "-i", f"{output_dir}/output.mp4", "-qscale:v", "2", f"{output_dir}/output2.mp4"])

        # add the audio back to the video
        subprocess.run(["ffmpeg", "-i", f"{output_dir}/output2.mp4", "-i", video_path, "-c", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", f"{output_dir}/output3.mp4"])

        yield sieve.Video(path=f'{output_dir}/output3.mp4')
