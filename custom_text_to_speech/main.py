import sieve
import soundfile as sf 

@sieve.Model(
    name="custom-tortoise-tts",
    python_packages=[
        "numpy==1.18.5",
        "requests==2.28.1",
        "librosa==0.9.2",
        "numba==0.56.4",
        "torch==1.8.1",
        "wget==3.2",
        "torchaudio==0.8.1"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libavcodec58"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth' -P /root/.cache/tortoise/models/",
        "wget -c 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth' -P /root/.cache/tortoise/models/",
        "pip install -e 'git+https://github.com/sieve-community/tortoise-tts.git@0cf6febaa52f67c3a8cd882330625606efa2b13d#egg=TorToiSe'"
    ],
    persist_output=True,
    iterator_input=True,
    gpu=True
   #machine_type='a100'
)
class TortoiseTTS:
    def __setup__(self):

        import torchaudio
        from tortoise.api import TextToSpeech, MODELS_DIR
        from tortoise.utils.audio import load_voices, load_audio
        from tortoise.utils.text import split_and_recombine_text
        import torch
        import os
        self.tts = TextToSpeech(models_dir=MODELS_DIR)
        print("running setup")

    def __predict__(self, text: str, preset: str, audio1: sieve.Audio, audio2: sieve.Audio) -> sieve.Audio:
        print("running predict")
        import time
        start_time=time.time()
        import torchaudio
        from tortoise.api import TextToSpeech, MODELS_DIR
        from tortoise.utils.audio import load_voices, load_audio
        from tortoise.utils.text import split_and_recombine_text
        import torch
        import os

        text, audio1, preset, audio2 = list(text)[0], list(audio1)[0], list(preset)[0], list(audio2)[0]
        all_parts=[]
        print("loading voice samples")
        if '.wav' not in audio1.path:
            print('Need a wav file')
        else:
            voice_samples = [load_audio(audio1.path, 22050), load_audio(audio2.path, 22050)]
        print("running tts with preset")
        latents = self.tts.get_conditioning_latents(voice_samples)
        gen = self.tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=latents,
                                preset=preset)
        gen = gen.squeeze(0).cpu()
        all_parts.append(gen)
        print("torch concat")
        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(os.path.join('combined.wav'), full_audio, 24000)
        print("Return")
        print(f"Time taken: {time.time() - start_time}")
        return sieve.Audio(path='combined.wav')