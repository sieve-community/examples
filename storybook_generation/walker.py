import sieve

@sieve.Model(
    name="run_stable_diff_walk",
    python_packages=[
        "torch==1.13.1",
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
    persist_output=True
)
class StableDiffusionVideo:
    def __setup__(self):
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        # Load stable diffusion model from local cache
        self.pipeline = StableDiffusionWalkPipeline.from_pretrained(
            "/root/.cache/models/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            revision="fp16",
        ).to("cuda")

        # Keep global ID to sort outputs
        self.video_number = 0
    
    def __predict__(self, prompt_pair: tuple) -> sieve.Video:
        import torch
        from stable_diffusion_videos import StableDiffusionWalkPipeline

        # Unpack prompt pair
        prompt1, prompt2 = prompt_pair[0], prompt_pair[1]

        # Generate and store video output
        print("Generating video with prompts: " + prompt1 + " | " + prompt2)
        video_path = self.pipeline.walk(
            [prompt1, prompt2],
            [42, 1337],
            fps=5,
            num_interpolation_steps=15,
            height=512,
            width=768,
        )

        # Increment global id
        self.video_number += 1

        # Return video
        yield sieve.Video(path=video_path, video_number=self.video_number)
