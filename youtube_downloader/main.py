import sieve

def on_progress(stream, chunk, bytes_remaining):
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    print(f'{percentage:.2f}% downloaded')

metadata = sieve.Metadata(
    description="Download highest-resolution version of YouTube video as an MP4.",
    code_url="https://github.com/sieve-community/examples/blob/main/youtube_downloader",
    image=sieve.Image(
        url="https://yt3.googleusercontent.com/584JjRp5QMuKbyduM_2k5RlXFqHJtQ0qLIPZpwbUjMJmgzZngHcam5JMuZQxyzGMV5ljwJRl0Q=s900-c-k-c0x00ffffff-no-rj"
    ),
    tags=["Video"],
    readme=open("README.md", "r").read(),
)

def merge_audio(video_with_audio, new_audio, output_video):
    import subprocess
    # Combine the new audio with the video
    merge_cmd = f"ffmpeg -y -i '{video_with_audio}' -i '{new_audio}' -c:v copy -map 0:v:0 -map 1:a:0 -shortest '{output_video}'"
    subprocess.call(merge_cmd, shell=True)

@sieve.function(
    name="youtube_to_mp4",
    system_packages=["ffmpeg"],
    python_packages=[
        "pytube @ git+https://github.com/sieve-community/pytube.git",
    ],
    metadata=metadata
)
def download(url: str, include_audio: bool = True):
    '''
    :param url: YouTube URL to download
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

    print("filtering stream for highest quality mp4...")
    video = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').desc().first()
    print('downloading video...')
    video.download(filename=video_filename)

    if include_audio:
        print('downloading audio...')
        audios = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        audios.download(filename=audio_filename)
        print('merging audio...')
        merge_audio(video_filename, audio_filename, "output.mp4")
        print('Done!')
        return sieve.File(path="output.mp4")
    else:
        print('Done!')
        return sieve.File(path=video_filename)

if __name__ == "__main__":
    download("https://www.youtube.com/watch?v=AKJfakEsgy0", include_audio=True)
