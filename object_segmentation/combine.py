import sieve


@sieve.function(
    name="frame-combiner",
    gpu=False,
    python_packages=[
        "opencv-python==4.6.0.66",
        "moviepy==1.0.3",
        "uuid==1.30",
        "ffmpeg-python==0.2.0",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    iterator_input=True,
    persist_output=True,
)
def frame_combine(it: sieve.Image) -> sieve.Video:
    import uuid
    import ffmpeg
    l = []
    for i in it:
        l.append(i)
        print(i.path, i.frame_number)
    sorted_by_frame_number = sorted(l, key=lambda k: k.frame_number)
    image_paths = [i.path for i in sorted_by_frame_number]

    if hasattr(l[0], 'fps'):
        fps = l[0].fps
    else:
        fps = 30

    video_path = f"{uuid.uuid4()}.mp4"
    process = ffmpeg.input('pipe:', r=str(fps), f='image2pipe').output(video_path, vcodec='libx264', pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    for in_file in image_paths:
        with open(in_file, 'rb') as f:
            jpeg_data = f.read()
            process.stdin.write(jpeg_data)

    process.stdin.close()
    process.wait()

    return sieve.Video(path=video_path)
