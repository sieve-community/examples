import sieve
import concurrent.futures

metadata = sieve.Metadata(
    title="Mediapipe Video Face Detection",
    description="Detect faces in a video with MediaPipe.",
    code_url="https://github.com/sieve-community/examples/tree/main/face_detection",
    image=sieve.Image(
        url="https://mediapipe.dev/images/mobile/face_detection_android_gpu.gif"
    ),
    tags=["Detection", "Video", "Face", "Featured"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="mediapipe_video_face_detector",
    system_packages=["ffmpeg"],
    metadata=metadata
)
def detector(video: sieve.Video):
    print("Starting video face detection...")
    video_path = video.path

    face_detector = sieve.function.get("sieve/mediapipe_face_detector")

    # use ffmpeg to extract frames from video
    import subprocess
    import os

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
    import time
    with concurrent.futures.ThreadPoolExecutor() as executor:
        count = 0
        for job in executor.map(face_detector.push, imgs):
            t = time.time()
            res = list(job.result())
            yield {
                "frame_number": count,
                "boxes": res[1]
            }
            print(f"Detected faces in frame {count} in {time.time() - t} seconds")
            count += 1

    # delete temp dir
    import shutil
    shutil.rmtree(temp_dir)

