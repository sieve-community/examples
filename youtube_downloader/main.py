import sieve
from typing import Literal
import re

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

def progress_hook(d):
    if d['status'] == 'downloading':
        percent_str = d.get('_percent_str', '0%')
        # remove ANSI color codes and convert to float
        percent = float(re.sub(r'\x1b\[[0-9;]*m', '', percent_str).replace('%', ''))
        if percent % 5 == 0 or percent == 100:
            print(f"Downloading... {percent:.1f}% complete")
@sieve.function(
    name="youtube_to_mp4",
    system_packages=["ffmpeg"],
    python_packages=["yt-dlp", "python-dotenv"],
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
    import json
    import random
    from dotenv import load_dotenv
    load_dotenv()

    output_filename = "output.mp4"

    if os.path.exists(output_filename):
        print("Deleting existing output file...")
        os.remove(output_filename)
    
    # load proxies from environment if exists
    proxies = []
    proxies_env = os.getenv('YOUTUBE_PROXIES')
    if proxies_env:
        try:
            proxies = json.loads(proxies_env)
        except json.JSONDecodeError:
            print("No proxies found in environment variable, proceeding without proxies...")
    
    if not proxies:
        print("No proxies found, proceeding without proxies...")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best',
        'outtmpl': output_filename,
        'quiet': True,  
        'no_warnings': True,
        'noprogress': True, 
        'progress_hooks': [progress_hook],
        'playlist_items': '1' ,
#        'verbose': True,
        'nopart': True,  # Add this line,


    }

    if resolution != "highest-available":
        if resolution == "lowest-available":
            ydl_opts['format'] = 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst'
        else:
            # try h264 codec first, then fallback to any codec
            ydl_opts['format'] = (
                f'bestvideo[height<={resolution[:-1]}][ext=mp4][vcodec^=avc1]+'
                f'bestaudio[ext=m4a]/'
                f'bestvideo[height<={resolution[:-1]}][ext=mp4]+'
                f'bestaudio[ext=m4a]/'
                f'best[height<={resolution[:-1]}][ext=mp4]/'
                f'best'
            )
    if not include_audio:
        ydl_opts['format'] = ydl_opts['format'].split('+')[0]

    max_retries = 6 if proxies else 1
    for attempt in range(max_retries):
        try:
            # try without proxy
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print('Download successful!')
            return sieve.File(path=output_filename)
        except Exception as e:
            print("Download failed, retrying...")
            if proxies:
                proxy = random.choice(proxies)
                ydl_opts['proxy'] = proxy
            else:
                print(f"Download failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempt{'s' if max_retries > 1 else ''}.")

    print('Done!')
    return sieve.File(path=output_filename)

if __name__ == "__main__":
   download("https://www.youtube.com/watch?v=AKJfakEsgy0") 