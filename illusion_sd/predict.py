import os
import torch
import random
from PIL import Image
from typing import List
from PIL.Image import LANCZOS
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
import sieve

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
BASE_CACHE = "model-cache"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
IMG_CACHE = "img-cache"
CACHE_DIR = "./hf-cache"


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

def resize_for_condition_image(input_image, width, height):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(min(width, height)) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img

metadata = sieve.Metadata(
    title="Illusion Diffusion HQ",
    description="Controlnet with Stable Diffusion Realistic Vision",
    code_url="https://github.com/sieve-community/examples/tree/main/illusion_sd",
    image=sieve.Image(
        url="https://storage.googleapis.com/mango-public-models/sv.png"
    ),
    tags=["Stable Diffusion", "Generative"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="stable-diffusion-controlnet",
    gpu=True,
    machine_type="a100",
    python_packages=[
        "diffusers==0.21.1",
        "torch==2.0.1",
        "ftfy==6.1.1",
        "scipy==1.9.3",
        "transformers==4.25.1",
        "accelerate==0.20.3",
        "xformers==0.0.21",
    ],
    cuda_version="11.8",
    metadata=metadata
)
class Controlnet:
    def __setup__(self):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster"
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            vae=vae,
            safety_checker=None,
        ).to("cuda")
        pipe.save_pretrained(CACHE_DIR, safe_serialization=True)
        """Load the model into memory to make running multiple predictions efficient"""
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
            cache_dir=VAE_CACHE,
        )
        self.controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            torch_dtype=torch.float16,
            cache_dir=CONTROL_CACHE,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=self.controlnet,
            vae=self.vae,
            safety_checker=None,
            torch_dtype=torch.float16,
            cache_dir=BASE_CACHE,
        ).to("cuda")


    # Define the arguments and types the model takes as input
    def __predict__(
        self,
        image: sieve.Image,
        prompt: str='beautiful mountain scene, 8k, fujifilm, masterpiece, detailed',
        negative_prompt: str='disfigured, ugly, low quality, blurry, nsfw',
        num_inference_steps: int=40,
        guidance_scale: float=7.5,
        seed: int=-1,
        width: int=768,
        height: int=768,
        controlnet_conditioning_scale: float=1.1,
    ) -> sieve.Image:
        seed = torch.randint(0, 2**32, (1,)).item() if seed == -1 else seed
        image = Image.open(str(image.path))
        out = self.pipe(
            prompt=[prompt],
            negative_prompt=[negative_prompt],
            image=[image],
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.Generator().manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )

        outputs = []
        for i, image in enumerate(out.images):
            fname = f"/tmp/output-{i}.png"
            image.save(fname)
            outputs.append(sieve.Image(path=fname))
        return outputs[-1]
