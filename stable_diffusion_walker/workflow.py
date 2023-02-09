'''
Sieve workflow to generate artsy stable diffusion videos morphing from one prompt to another.

Community model URL: https://www.sievedata.com/dashboard/templates/au-sievedata-com/stable_diffusion_walker
Source: https://github.com/nateraw/stable-diffusion-videos
'''

import sieve

@sieve.Model(
    name="run_stable_diff_walk",
    python_packages=[
        "torch==1.8.1",
        "stable_diffusion_videos==0.8.1",
        "accelerate==0.16.0"
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
    persist_output=True
)
class StableDiffusionVideo:
    def __setup__(self):
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        # load stable diffusion model from local cache
        self.pipeline = StableDiffusionWalkPipeline.from_pretrained(
            "/root/.cache/models/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            revision="fp16",
        ).to("cuda")
    
    def __predict__(self, prompt1: str, prompt2: str) -> sieve.Video:
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        prompt1, prompt2 = list(prompt1)[0], list(prompt2)[0] # current workaround for iterator inputs

        # generate and store video output
        video_path = self.pipeline.walk(
            [prompt1, prompt2],
            [42, 1337],
            fps=5,
            num_interpolation_steps=15,
            height=512,
            width=512,
        )

        return sieve.Video(path=video_path)

@sieve.workflow(name="stable_diffusion_walker")
def stable_diffusion_walker(from_prompt: str, to_prompt: str) -> sieve.Video:
    return StableDiffusionVideo()(from_prompt, to_prompt)