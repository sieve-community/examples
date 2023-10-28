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

@sieve.function(
    name="youtube_to_mp4",
    python_packages=[
        "pytube @ git+https://github.com/sieve-community/pytube"
    ],
    metadata=metadata
)
def download(url: str, include_audio: bool = True):
    '''
    :param url: YouTube URL to download
    :param include_audio: whether or not to ensure the audio is included in the video. if set to false, audio may still be included on occassion.
    :return: The downloaded YouTube video
    '''
    from pytube import YouTube
    import os

    output_filename = "output.mp4"
    
    if os.path.exists(output_filename):
        print("deleting temp file...")
        os.remove(output_filename)

    print("setting stream...")
    yt = YouTube(url)
    yt.register_on_progress_callback(on_progress)
    print("filtering stream for highest quality mp4...")
    streams = yt.streams.filter(progressive=include_audio, file_extension='mp4').order_by('resolution').desc()
    highest_res_stream = streams.first()

    print('downloading video...')
    highest_res_stream.download(filename=output_filename)

    print("finished downloading video, returning...")
    return sieve.Video(path=output_filename)