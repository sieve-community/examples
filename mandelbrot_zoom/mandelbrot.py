import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.cm
import sieve
from typing import List, Iterator, Dict

"""
# Commented out because custom types as workflow inputs doesn't work
class MandelbrotZoomConfig(sieve.Struct):
    n_frames: int # number of frames
    width: int # dimensions of output
    height: int
    z_real: float # point to zoom into (these are floats instead of complex because 
    z_imag: float
    initial_zoom: float # initial width of frame on the argand plane
    zoom_ratio: float # amount to zoom in by end of video
    max_iter: int # number of iterations to run
"""

def default_config():
    real = -0.92233278103709470 # cool mini mandelbrot point
    im = 0.3102598350874
    return sieve.Struct(n_frames=200, width=720, height=480, z_real=real, z_imag=im, initial_zoom=4., zoom_ratio=700., max_iter=80)

def mandelbrot_iter(z, max_iter):
    c = z
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(x_min, x_max, y_min, y_max, width, height, max_iter):
    r1 = np.linspace(x_min, x_max, width)
    r2 = np.linspace(y_min, y_max, height)
    n3 = np.empty((height, width, 3))
    colormap = matplotlib.cm.get_cmap("viridis").colors
    for i in range(width):
        for j in range(height):
            n3[height-j-1, i] = 256*np.array(colormap[int(mandelbrot_iter(r1[i] + 1j*r2[j], max_iter)/max_iter*(len(colormap)-1))])
    return n3

@sieve.function(name="frame-gen", persist_output=False, gpu=False, python_version="3.8")
def frame_generator(info: sieve.Struct):
    for i in range(info.n_frames):
        r = info.zoom_ratio**(i/(info.n_frames-1))
        pad_w = info.initial_zoom / r / 2.
        pad_h = pad_w * info.height/info.width
        yield sieve.Struct(i=i, xmin=info.z_real-pad_w, xmax=info.z_real+pad_w, ymin=info.z_imag-pad_h, ymax=info.z_imag+pad_h, width=info.width, height=info.height, max_iter=info.max_iter)

@sieve.function(name="mandelbrot", persist_output=False, gpu=False, python_packages=["matplotlib==3.6.3"], python_version="3.8")
def mandelbrot(x: sieve.Struct) -> sieve.Image:
    return sieve.Image(frame_number=x.i, array=mandelbrot_set(x.xmin, x.xmax, x.ymin, x.ymax, x.width, x.height, x.max_iter))

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
def mandelbrot_wf(info: sieve.Struct) -> sieve.Video:
    return frame_combine(mandelbrot(frame_generator(info)))


if __name__ == "__main__":
    sieve.upload(frame_generator)
    sieve.upload(mandelbrot)
    sieve.upload(frame_combine)
    sieve.deploy(mandelbrot_wf)

    sieve.push(mandelbrot_wf, inputs={"info": default_config()})

