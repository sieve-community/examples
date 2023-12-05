import sieve
import subprocess
import re

def get_video_resolution(file_path):
    # Run ffmpeg to get video info
    command = ["ffmpeg", "-i", file_path]
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process.communicate()

    # Use regular expression to find the resolution
    matches = re.search(r'Stream.*Video:.* (\d+)x(\d+)', err.decode('utf-8'), re.IGNORECASE)
    if matches:
        width, height = map(int, matches.groups())
        return width, height
    else:
        raise ValueError("Could not find resolution information")

metadata = sieve.Metadata(
    description="Upscale an input video by a given factor.",
    code_url="https://github.com/sieve-community/examples/blob/main/upscaling/",
    image=sieve.Image(
        url="https://assets-global.website-files.com/6005fac27a49a9cd477afb63/646374beaa7aae1186a82817_tpai-orange-tiger-after.jpg"
    ),
    tags=["Video", "Upscaling"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="video_upscaling",
    system_packages=["ffmpeg"],
    metadata=metadata
)
def upscaler(
    video: sieve.Video,
    scale: float = 2
):
    '''
    :param video: the video the upscale.
    :param scale: upscale the video by a factor of scale. Default is 2.
    :return: Upscaled video
    '''

    if scale > 5:
        raise ValueError("Max allowed uspcaling factor is 5")

    width, height = get_video_resolution(video.path)
    print(f"Width: {width}, Height: {height}")
    # if the video is more than 1440p, then reject.
    if width > 2560 or height > 1440:
        raise ValueError("Video resolution is more than 1440p, rejecting as upscaling doesn't make quite a different in this case.")


    # split video into 10 second chunks
    import subprocess
    import os
    import tempfile
    import shutil

    temp_dir = os.path.join(os.getcwd(), 'tmp')
    # if directory exists, delete it
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    # create temporary directory
    os.makedirs(temp_dir, exist_ok=True)

    # split video into 5 second chunks
    subprocess.run([
        "ffmpeg",
        "-i", video.path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", "00:00:02",
        "-f", "segment",
        "-reset_timestamps", "1",
        "-y",
        os.path.join(temp_dir, "chunk-%05d.mp4")
    ])

    # upscale each chunk
    esrgan = sieve.function.get("sieve/real-esrgan")
    upscaled_chunks = []
    count = 0
    for chunk in sorted(os.listdir(temp_dir)):
        print(f"Sending chunk {count} for upscaling")
        j = esrgan.push(
            sieve.File(path=os.path.join(temp_dir, chunk)),
            scale=scale
        )
        upscaled_chunks.append(j)
        count += 1

    # concatenate chunks as they finish
    out_chunk = None
    paths = []
    count = 0
    for j in upscaled_chunks:
        out_chunk = j.result()
        print(f"Chunk {count} finished upscaling")
        paths.append(out_chunk.path)
        count += 1

    # add concatenation logic here
    # concatenate all chunks
    concat_command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i"]
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".txt") as f:
        f.write('\n'.join("file '" + path + "'" for path in paths).encode())
        concat_command.append(f.name)
        concat_command.extend(["-c", "copy", "upscaled.mp4"])
    subprocess.run(concat_command)
    out_path = "upscaled.mp4"

    # cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return sieve.Video(path=out_path)

if __name__=="__main__":
    vid = sieve.Video(path="/Users/Mokshith/Downloads/d8fce3bf-52b9-4012-94af-f80e57d1b7e3-input-source_video.mp4")
    print(upscaler(vid))