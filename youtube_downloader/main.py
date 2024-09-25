import sieve
from typing import Literal

metadata = sieve.Metadata(
    title = "YouTube Video Downloader",
    description="Download YouTube videos in MP4 format at any resolution.",
    code_url="https://github.com/sieve-community/examples/blob/main/youtube_downloader",
    image=sieve.Image(
        url="https://yt3.googleusercontent.com/584JjRp5QMuKbyduM_2k5RlXFqHJtQ0qLIPZpwbUjMJmgzZngHcam5JMuZQxyzGMV5ljwJRl0Q=s900-c-k-c0x00ffffff-no-rj"
    ),
    tags=["Video"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="youtube_to_mp4",
    system_packages=["ffmpeg"],
    python_packages=["yt-dlp"],
    metadata=metadata,
)
def download(
    url: str,
    resolution: Literal["highest-available", "lowest-available", "1080p", "720p", "480p", "360p", "240p", "144p"] = "highest-available",
    include_audio: bool = True,
):
    '''
    :param url: YouTube URL to download
    :param resolution: The resolution of the video to download. If the desired resolution is not available, the closest resolution will be downloaded instead.
    :param include_audio: Whether to include audio in the video.
    :return: The downloaded YouTube video
    '''
    import yt_dlp
    import os

    output_filename = "output.mp4"

    if os.path.exists(output_filename):
        print("Deleting existing output file...")
        os.remove(output_filename)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best',
        'outtmpl': output_filename,
    }

    if resolution != "highest-available":
        if resolution == "lowest-available":
            ydl_opts['format'] = 'worstvideo[ext=mp4][vcodec^=avc1]+worstaudio[ext=m4a]/worst[ext=mp4][vcodec^=avc1]/worst'
        else:
            ydl_opts['format'] = f'bestvideo[height<={resolution[:-1]}][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<={resolution[:-1]}][ext=mp4][vcodec^=avc1]/best'

    if not include_audio:
        ydl_opts['format'] = ydl_opts['format'].split('+')[0]

    print("Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print('Done!')
    return sieve.File(path=output_filename)

if __name__ == "__main__":
    path  = 'video_144p_audio_True.mp4'
    import subprocess
    import json

    # Function to get video codec using ffprobe
    def get_video_codec(file_path):
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return stream['codec_name']
        
        return None

    # Check the codec of the video
    video_codec = get_video_codec(path)
    if video_codec:
        print(f"The video codec is: {video_codec}")
    else:
        print("Could not determine the video codec.")