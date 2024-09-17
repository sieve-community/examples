import sieve
from typing import Literal
def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    print(f'{percentage:.2f}% downloaded')

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

def merge_audio(video_with_audio, new_audio, output_video, convert_codec = False):
    import subprocess
    # Combine the new audio with the video
    if convert_codec:
        merge_cmd = f"ffmpeg -y -i '{video_with_audio}' -i '{new_audio}' -c:v copy -map 0:v:0 -map 1:a:0 -shortest '{output_video}'"
    else:
        merge_cmd = f"ffmpeg -y -i '{video_with_audio}' -i '{new_audio}' -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest '{output_video}'"
    subprocess.call(merge_cmd, shell=True)

@sieve.function(
    name="youtube_to_mp4",
    system_packages=["ffmpeg"],
    python_packages=[
        "pytube @ git+https://github.com/sieve-community/pytube.git",
    ],
    metadata=metadata
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
    from pytube import YouTube
    import os

    video_filename = "video.mp4"
    audio_filename = "audio.mp3"
    
    if os.path.exists(video_filename):
        print("deleting temp video file...")
        os.remove(video_filename)
    
    if os.path.exists(audio_filename):
        print("deleting temp audio file...")
        os.remove(audio_filename)

    print("setting stream...")
    yt = YouTube(url)
    yt.register_on_progress_callback(on_progress)

    #print("filtering stream for highest quality mp4...")
    all_streams = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').desc() #.first()
    video = [stream for stream in all_streams if stream.video_codec.startswith('avc1')]
    if resolution == "highest-available":
        video = video[0]
        print(f"highest available resolution is {video.resolution}...")
    elif resolution == "lowest-available":
        video = video[-1]
        print(f"lowest available resolution is {video.resolution}...")
    else:
        desired_res = int(resolution.replace('p', ''))
        diff_list = [(abs(desired_res - int(stream.resolution.replace('p', ''))), stream) for stream in video]
        diff_list.sort(key=lambda x: x[0])
        video = diff_list[0][1]
        if video.resolution != resolution:
            print(f"{resolution} resolution is not available, using {video.resolution} instead...")
        else:
            print(f"selected resolution is {resolution}...")

        
    print('downloading video...')
    video.download(filename=video_filename)

    if include_audio:
        print('downloading audio...')
        audios = yt.streams.filter(only_audio=True, mime_type = "audio/mp4").order_by('abr').desc().first()
        convert_codec = False
        if not audios:
            print("No audio stream found for mp4, downloading audio and converting...")
            audios = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
            convert_codec = True

        audios.download(filename=audio_filename)
        print('merging audio...')
        merge_audio(video_filename, audio_filename, "output.mp4", convert_codec)
        print('Done!')
        return sieve.File(path="output.mp4")
    else:
        print('Done!')
        return sieve.File(path=video_filename)



if __name__ == "__main__":
    download("https://www.youtube.com/watch?v=AKJfakEsgy0", include_audio=True)