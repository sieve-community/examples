import sieve

@sieve.function(
    name="video-combiner",
    gpu = False,
    python_packages=[
        "moviepy==1.0.3",
        "uuid==1.30",
    ],
    python_version="3.8",
    iterator_input=True,
    persist_output=True
)
def combiner(videos) -> sieve.Video:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    import uuid

    # Sort videos by global id
    videos = sorted(videos, key=lambda video: video.video_number)

    # Combine videos
    videos = [VideoFileClip(video.path) for video in videos]
    video = concatenate_videoclips(videos)

    # Save video
    video_path = f"{uuid.uuid4()}.mp4"
    video.write_videofile(video_path)
    return sieve.Video(path=video_path)
    