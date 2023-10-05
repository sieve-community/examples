import sieve

metadata = sieve.Metadata(
    description="Generate audio that sounds similar to a reference speaker in 13 languages",
    tags=["Audio", "Speech", "TTS"],
    image=sieve.Image(
        url="https://miro.medium.com/v2/resize:fit:512/0*bWnCJKvDb1wmPiPP"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="xtts-v1",
    machine_type="a100",
    gpu=True,
    python_packages=[
        "git+https://github.com/coqui-ai/TTS.git"
    ],
    run_commands=[
        "mkdir -p /root/.cache/model",
        "git lfs install",
        "git lfs clone https://huggingface.co/coqui/XTTS-v1 /root/.cache/model"
    ],
    system_packages=[
        "libsndfile1",
        "ffmpeg",
        "git-lfs",
    ],
    python_version="3.11",
    cuda_version="11.8",
    metadata=metadata
)
class XTTS:
    def __setup__(self):
        import torch
        import os

        from TTS.api import TTS
        os.environ["COQUI_TOS_AGREED"] = "1"

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        from scipy.io import wavfile

        self.config = XttsConfig()
        self.config.load_json("/root/.cache/model/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir="/root/.cache/model/", eval=True)
        self.model.cuda()

    def __predict__(self, reference_speaker_audio: sieve.Audio, language: str, text: str) -> sieve.Audio:
        """
        :param reference_speaker_audio: an audio sample of the original speaker voice to copy
        :param language: language of the text to generate. try "en" for English, "es" for Spanish, or others below.
        :param text: text to generate (must match the language)
        :return: Generated audio
        """
        from scipy.io import wavfile
        import os
        import torchaudio

        speaker_audio_path = reference_speaker_audio.path
        # Convert to wav if needed
        extension = speaker_audio_path.split(".")[-1].lower()
        if extension != "wav":
            waveform, sample_rate = torchaudio.load(speaker_audio_path)
            if os.path.exists("speaker.wav"):
                os.remove("speaker.wav")
            torchaudio.save("speaker.wav", waveform, sample_rate)
            speaker_audio_path = "speaker.wav"

        if len(text) < 2:
            raise ValueError("Please give a longer prompt text")
        
        output = self.model.synthesize(text,
            self.config,
            speaker_wav=speaker_audio_path,
            language=language,
        )

        if os.path.exists("output.wav"):
            os.remove("output.wav")

        wavfile.write("output.wav", self.config.audio.sample_rate, output['wav'])
        return sieve.Audio(path="output.wav")
