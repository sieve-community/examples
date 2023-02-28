import sieve

@sieve.function(
    name="video-captioner",
    gpu = False,
    python_packages=[
        "moviepy==1.0.3",
        "uuid==1.30",
    ],
    system_packages=["imagemagick"],
    run_commands=["apt install -y imagemagick"],
    iterator_input=True,
    persist_output=True
)
def captioner(videos, prompt_pair) -> sieve.Video:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    import uuid

    for v, prompt in zip(videos, prompt_pair):
        video = VideoFileClip(v.path)

        # Create caption
        prompt = prompt[0]
        middle = len(prompt) // 2
        caption = prompt[:middle] + "\n" + prompt[middle:]
        text = TextClip(caption, font='calibri', fontsize=24, color='white')
        text = text.set_pos('bottom').set_duration(video.duration)

        # Combine video and caption
        video = CompositeVideoClip([video, text])
        video.write_videofile("bear_with_text.mp4")

        # Save video
        video_path = f"{uuid.uuid4()}.mp4"
        video.write_videofile(video_path)
        return sieve.Video(path=video_path)
    