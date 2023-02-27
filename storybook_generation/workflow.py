'''
Storybook generation workflow
'''

import sieve
from walker import StableDiffusionVideo
from combiner import frame_combine
from splitter import VideoSplitter

# TODO: make this use an LLM
@sieve.function(name="prompt-to-script")
def prompt_to_script(prompt: str) -> list:
    script = prompt.split(".")
    script = [s.strip() for s in script if s.strip() != ""]
    script = [s + "." for s in script]
    return script

@sieve.function(
    name="script-to-video",
    python_packages=[
        "torch==1.8.1",
        "stable_diffusion_videos==0.8.1",
        "accelerate==0.16.0",
        "opencv-python==4.6.0.66",
        "moviepy==1.0.3",
        "uuid==1.30",
        "ffmpeg-python==0.2.0",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libavcodec58", "libsndfile1", "git-lfs"],
    gpu=True,
    machine_type="a100",
    run_commands=[
        "mkdir -p /root/.cache/models/stable-diffusion-v1-4",
        "git lfs install",
        "git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 /root/.cache/models/stable-diffusion-v1-4",
    ],
    iterator_input=True,
    persist_output=True,
    python_version="3.8",
)
def script_to_video(script: list) -> sieve.Video:
    images = []
    for i in range(len(script) - 1):
        prompt1, prompt2 = script[i], script[i + 1]
        video = StableDiffusionVideo()(prompt1, prompt2)
        images.append(VideoSplitter(video))
    return frame_combine(images)

@sieve.workflow(name="storybook_generation")
def storybook_generation(prompt: str) -> sieve.Video:
    script = prompt_to_script(prompt)
    video = script_to_video(script)
    return video
