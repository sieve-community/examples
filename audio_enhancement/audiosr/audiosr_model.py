import sieve
import os
import random

import numpy as np
import soundfile as sf
import torch
import shutil
import yaml

model_metadata = sieve.Metadata(
    description='AudioSR: Versatile Audio Super-resolution at Scale',
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement/audiosr/audiosr_model.py",
    image=sieve.Image(
        url="https://github.com/haoheliu/versatile_audio_super_resolution/raw/main/visualization.png?raw=true"
    ),
    tags=["Audio", "Speech", "Enhancement"],
    readme=open("AUDIOSR_README.md", "r").read(),
)

@sieve.Model(
    name="audioSR",
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
        "wget -P /root/.cache/models/ https://huggingface.co/haoheliu/wellsolve_audio_super_resolution_48k/resolve/main/basic.pth"
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

        ckpt_path = '/root/.cache/models/basic.pth'
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
        '''
        :param audio: audio to enhance
        :param ddim_steps: number of diffusion steps
        :param guidance_scale: scale of guidance
        :return: audio upsampled to 48kHz
        '''

        from audiosr import super_resolution

        seed = random.randint(0, 2**32 - 1)
        print(f"Setting seed to: {seed}")

        waveform = super_resolution(
            self.model,
            audio.path,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            latent_t_per_second=12.8,
        )
        out_wav = (waveform[0] * 32767).astype(np.int16).T
        sf.write("out.wav", data=out_wav, samplerate=48000)
        return sieve.Audio(path="out.wav")
