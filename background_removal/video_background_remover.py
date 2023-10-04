#TODO: do stuff

import sieve
import concurrent.futures

@sieve.function(
    name="video_background_remover",
    system_packages=["ffmpeg"],
    python_packages=["opencv-python"]
)
def remove(video: sieve.Video) -> sieve.Video:
    video_path = video.path

    DIS_background_remover = sieve.function.get("sieve/dis")

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
        print(f"Frame {frame_num} created for background remover")
    
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
        for job in executor.map(DIS_background_remover.push, imgs):
            t = time.time()
            for res in job.result():
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

if __name__=="__main__":
    remove.run(sieve.Video(path="/Users/Mokshith/Desktop/lebron-miami1.mp4"))