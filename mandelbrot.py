import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm
import sieve
from typing import List, Iterator, Dict

def mandelbrot(z, max_iter):
    c = z
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iter):
    r1 = np.linspace(x_min, x_max, width)
    r2 = np.linspace(y_min, y_max, height)
    n3 = np.empty((width, height, 3))
    colormap = matplotlib.cm.get_cmap("viridis").colors
    for i in range(width):
        for j in range(height):
            n3[height-j-1, i] = 256*np.array(colormap[int(mandelbrot(r1[i] + 1j*r2[j], max_iter)/max_iter*(len(colormap)-1))])
    return n3

def mandelbrot_image(x_min, x_max, y_min, y_max, width=1024, height=1024, max_iter=80):
    img_width = width
    img_height = height
    return mandelbrot_set(x_min, x_max, y_min, y_max, img_width, img_height, max_iter)

@sieve.function(name="mandelbrot", persist_output=False, gpu=False, python_packages=["matplotlib==3.6.3"], python_version="3.8")
def mandelbrotfn(x: sieve.Struct) -> sieve.Image:
    return sieve.Image(frame_number=x.i, array=mandelbrot_image(x.xmin, x.xmax, x.ymin, x.ymax))

@sieve.function(name="gen", persist_output=False, gpu=False, python_version="3.8")
def frame_generator(info: sieve.Struct):
    for i in range(info.n):
        padding = info.pad * (info.r ** i)
        yield sieve.Struct(i=i, xmin=info.real-padding, xmax=info.real+padding, ymin=info.im-padding, ymax=info.im+padding)

#−0.9223327810370947027656057193752719757635 +0.3102598350874576432708737495917724836010i 


@sieve.function(
    name="frame-combiner",
    gpu = False,
    python_packages=[
        "opencv-python==4.6.0.66",
        "moviepy==1.0.3",
        "uuid==1.30",
        "ffmpeg-python==0.2.0"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    iterator_input=True,
    persist_output=True
)
def frame_combine(it: sieve.Image) -> sieve.Video:
    import uuid
    import ffmpeg
    import time
    l = []
    for i in it:
        l.append(i)
        print(i.path, i.frame_number)
    sorted_by_frame_number = sorted(l, key=lambda k: k.frame_number)
    image_paths = [i.path for i in sorted_by_frame_number]

    # use ffmpeg to combine frames
    video_path = f"{uuid.uuid4()}.mp4"
    process = ffmpeg.input('pipe:', r='20', f='image2pipe').output(video_path, vcodec='libx264', pix_fmt='yuv420p').overwrite_output().run_async(pipe_stdin=True)
    # Iterate jpeg_files, read the content of each file and write it to stdin
    for in_file in image_paths:
        with open(in_file, 'rb') as f:
            data = f.read()
            process.stdin.write(data)

    # Close stdin pipe - FFmpeg fininsh encoding the output file.
    process.stdin.close()
    process.wait()

    return sieve.Video(path=video_path)

@sieve.workflow(name="mandelbrot-wf")
def mandelbrot_wf(info: sieve.Struct) -> sieve.Image:
    return frame_combine(mandelbrotfn(frame_generator(info)))

real = -0.92233278103709470
im = 0.3102598350874

if __name__ == "__main__":
    #sieve.upload(mandelbrotfn)
    #sieve.upload(frame_combine)
    #sieve.upload(frame_generator)
    #sieve.upload(vid_stitcher)
    #sieve.deploy(mandelbrot_wf)

    x = sieve.Struct(real=real, im=im, pad=1.5, n=200, r=.97)
    sieve.push(mandelbrot_wf, inputs={"info": x})

