import sieve
import os
import random

import numpy as np
import soundfile as sf
import torch
import librosa
import subprocess
import os

model_metadata = sieve.Metadata(
    description="AudioSR: Versatile Audio Super-resolution at Scale",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement/audiosr/audiosr_model.py",
    image=sieve.Image(
        url="https://github.com/haoheliu/versatile_audio_super_resolution/raw/main/visualization.png?raw=true"
    ),
    tags=["Audio", "Speech", "Enhancement"],
    readme=open("AUDIOSR_README.md", "r").read(),
)


@sieve.Model(
    name="audiosr",
    gpu="a100",
    python_packages=[
        "torch==1.13.1",
        "torchaudio",
        "torchvision",
        "tqdm==4.66.1",
        "gradio==3.44.4",
        "pyyaml==6.0.1",
        "einops==0.6.1",
        "chardet==5.2.0",
        "numpy==1.23.5",
        "soundfile==0.12.1",
        "librosa==0.9.2",
        "scipy==1.11.2",
        "pandas==2.1.0",
        "unidecode==1.3.6",
        "phonemizer==3.2.1",
        "torchlibrosa==0.1.0",
        "transformers==4.30.2",
        "huggingface_hub==0.17.2",
        "progressbar==2.5",
        "ftfy==6.1.1",
        "timm==0.9.7",
        "audiosr==0.0.5",
    ],
    system_packages=[
        "ffmpeg",
        "libsndfile1",
    ],
    python_version="3.9",
    run_commands=[
        "pip install numpy==1.23.5 --force-reinstall",
        "wget -P /root/.cache/models/ https://huggingface.co/haoheliu/wellsolve_audio_super_resolution_48k/resolve/main/basic.pth",
    ],
    metadata=model_metadata,
)
class AudioSr:
    def __setup__(self):
        from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
        from audiosr.utils import default_audioldm_config

        self.model_name = "basic"
        self.device = torch.device("cuda:0")
        self.sr = 48000

        print("Loading AudioSR: %s" % self.model_name)
        print("Loading model on %s" % self.device)

        ckpt_path = "/root/.cache/models/basic.pth"
        config = default_audioldm_config(self.model_name)
        config["model"]["params"]["device"] = self.device

        latent_diffusion = LatentDiffusion(**config["model"]["params"])
        resume_from_checkpoint = ckpt_path

        checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
        latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
        latent_diffusion.eval()
        latent_diffusion = latent_diffusion.to(self.device)

        self.model = latent_diffusion

    def __predict__(
        self, audio: sieve.Audio, ddim_steps: int = 50, guidance_scale: float = 3.5
    ) -> sieve.Audio:
        """
        :param audio: audio to enhance (mp3 and wav)
        :param ddim_steps: number of diffusion steps
        :param guidance_scale: scale of guidance
        :return: audio upsampled to 48kHz
        """

        from audiosr import super_resolution

        # Delete combined.wav if it exists
        if os.path.exists("combined.wav"):
            os.remove("combined.wav")

        seed = random.randint(0, 2**32 - 1)
        print(f"Setting seed to: {seed}")

        audio_path = audio.path
        audio_dir = os.path.dirname(audio_path)
        audio_base = os.path.basename(audio_path)
        audio_name, audio_ext = os.path.splitext(audio_base)

        chunk_dir = os.path.join(audio_dir, f"{audio_name}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_path = os.path.join(chunk_dir, f"{audio_name}_chunk%03d{audio_ext}")

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            "10",
            "-c",
            "copy",
            chunk_path,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error occurred while running the command:")
            print(e.stderr.decode())
            exit(1)

        print(f"Audio chunks saved in: {chunk_dir}")

        chunk_files = sorted(os.listdir(chunk_dir))
        chunk_files = [
            os.path.join(chunk_dir, file)
            for file in chunk_files
            if file.endswith(audio_ext)
        ]

        mean, std = None, None
        for idx, chunk_file in enumerate(chunk_files):
            orig_chunk, _ = librosa.load(chunk_file, sr=48000)
            print(f"Original chunk shape: {orig_chunk.shape}")

            trimmed_audio, ys = librosa.effects.trim(
                orig_chunk, top_db=75, frame_length=256, hop_length=64
            )
            if idx == 0:
                trimmed_audio = orig_chunk[: ys[1]]
                mean = np.mean(orig_chunk)
                std = np.max(np.abs(orig_chunk)) + 1e-8
            elif idx == len(chunk_files) - 1:
                trimmed_audio = orig_chunk[ys[0] :]
            print("Trimmed chunk shape: ", trimmed_audio.shape)

            waveform = super_resolution(
                self.model,
                chunk_file,
                seed=seed,
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                latent_t_per_second=12.8,
                mean=mean,
                std=std,
            )
            chunk = (waveform[0] * 32767).astype(np.int16).T

            print("Enhanced chunk shape: ", chunk.shape)
            chunk = chunk[ys[0] : ys[1]]
            print("Enhanced chunk shape after cropping: ", chunk.shape)
            chunk_file_path = os.path.join(chunk_dir, f"enhanced_{idx}.wav")
            sf.write(chunk_file_path, data=chunk, samplerate=48000)

        # Get all enhanced chunk files
        enhanced_chunk_files = sorted(os.listdir(chunk_dir))
        enhanced_chunk_files = [
            os.path.join(chunk_dir, file)
            for file in enhanced_chunk_files
            if "enhanced" in file
        ]

        with open("files.txt", "w") as f:
            for file in enhanced_chunk_files:
                f.write(f"file '{file}'\n")

        combined_command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "files.txt",
            "combined.wav",
        ]
        try:
            result = subprocess.run(
                combined_command, check=True, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print("Error occurred while running the command:")
            print(e.stderr.decode())
            exit(1)

        subprocess.run(["rm", "-rf", chunk_dir], check=True)
        subprocess.run(["rm", "files.txt"], check=True)

        return sieve.Audio(path="combined.wav")
