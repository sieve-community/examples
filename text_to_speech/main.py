import sieve

@sieve.Model(
    name="tortoise-tts",
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
    gpu=True,
    machine_type='a100'
)
class TortoiseTTS:
    def __setup__(self):
        import torchaudio
        from tortoise.api import TextToSpeech, MODELS_DIR
        from tortoise.utils.audio import load_voices, load_audio
        from tortoise.utils.text import split_and_recombine_text
        import torch
        self.tts = TextToSpeech(models_dir=MODELS_DIR)

    def __predict__(self, text: str) -> sieve.Audio:
        import torchaudio
        from tortoise.api import TextToSpeech, MODELS_DIR
        from tortoise.utils.audio import load_voices, load_audio
        from tortoise.utils.text import split_and_recombine_text
        import torch
        import os

        if '|' in text:
            print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
                "your intent, please remove all '|' characters from the input.")
            texts = text.split('|')
        else:
            texts = split_and_recombine_text(text)

        voice_sel = ['freeman']
        voice_samples, conditioning_latents = load_voices(voice_sel)
        all_parts = []
        for j, text in enumerate(texts):
            gen = self.tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                    preset="ultra_fast", k=1, use_deterministic_seed=1)
            gen = gen.squeeze(0).cpu()
            all_parts.append(gen)

        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(os.path.join('combined.wav'), full_audio, 24000)
        return sieve.Audio(path='combined.wav')
    
