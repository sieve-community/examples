import sieve

metadata = sieve.Metadata(
    description="Generate speech, sound effects, music and beyond.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_generation/audioldm",
    image=sieve.Image(
        url="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYWeO-CJufmSOlSyrqxFxK6dWRM1l7tqMHzA&usqp=CAU"
    ),
    tags=["Audio", "Speech"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="audioldm",
    python_version="3.9",
    cuda_version="11.8",
    gpu=sieve.gpu.T4(),
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_packages=[
        "torch==2.1.0",
        "scipy",
        "git+https://github.com/huggingface/diffusers.git",
        "transformers",
        "accelerate"
    ],
    metadata=metadata
)
class AudioLDM:
    def __setup__(self):
        from diffusers import AudioLDM2Pipeline
        import torch

        repo_id = "cvssp/audioldm2"
        self.pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        from diffusers import DPMSolverMultistepScheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def __predict__(
        self,
        prompt: str,
        duration: float = 5.0,
        num_inference_steps: int = 20,
        negative_prompt: str = "Low quality, average quality."
    ) -> sieve.File:
        '''
        :param prompt: The prompt to generate audio from.
        :param duration: The duration of the audio in seconds.
        :param num_inference_steps: The number of inference steps to run.
        :param negative_prompt: The negative prompt to use.
        :return: The generated audio.
        '''
        settings = f'''Generation Settings:
        Prompt: {prompt}
        Duration: {duration} seconds
        Number of Inference Steps: {num_inference_steps}
        Negative Prompt: {negative_prompt}
        '''
        print(settings)
        print("Generating audio...")
        import scipy.io.wavfile

        audio = self.pipe(prompt, num_inference_steps=num_inference_steps, audio_length_in_s=duration, negative_prompt=negative_prompt).audios[0]

        audio_path = f'/tmp/techno.wav'
        scipy.io.wavfile.write(audio_path, rate=16000, data=audio)
        return sieve.File(path=audio_path)
