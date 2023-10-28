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
    metadata=metadata,
    gpu=True,
    python_packages=[
        "git+https://github.com/coqui-ai/TTS.git@v0.19.1",
        "transformers",
    ],
    run_commands=[
        "mkdir -p /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1/model.pth -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1/config.json -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1/vocab.json -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v1/v1.1/hash.md5 -q -P /root/.cache/model"
    ],
    system_packages=[
        "libsndfile1",
        "ffmpeg",
        "git-lfs",
    ],
    python_version="3.11",
    cuda_version="11.8"
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

    def __predict__(
            self,
            text: str,
            reference_audio: sieve.Audio,
            stability: float = 0.5,
            similarity_boost: float = 0.63,
            language_code: str = "",
        ) -> sieve.Audio:
        """
        :param text: text to speak
        :param reference_audio: audio of the reference speaker
        :param stability: Value between 0 and 1. Increasing variability can make speech more expressive with output varying between re-generations. It can also lead to instabilities.
        :param similarity_boost: Value between 0 and 1. Low values are recommended if background artifacts are present in generated speech.
        :param language_code: language code of the text to generate. try "en" for English, "es" for Spanish, or others below. If left blank, the language will be detected automatically.
        :return: Generated audio
        """

        from scipy.io import wavfile
        import os
        import torchaudio

        import sieve

        langid = sieve.function.get("sieve/langid")
        langid_output = langid.push(text)

        speaker_audio_path = reference_audio.path
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

        langid_output = langid_output.result()
        if len(language_code) > 0 and langid_output["language_code"].strip() != language_code:
            print(f"Warning: language code mismatch. You specified {language_code} but langid detected {langid_output['language_code']}, attempting to continue...")
            language = language_code
        else:
            language = langid_output["language_code"].strip()
        
        if language == "zh":
            # xtts only supports zh-cn
            language = "zh-cn"

        # Adjust token sampling: tighter sampling for higher similarity_boost for more focused generation
        top_k = 50 - int(similarity_boost * 20)
        top_p = 1.0 - similarity_boost * 0.15

        # Adjust repetition penalty: higher values for higher similarity_boost to prevent repetition
        repetition_penalty = 1.0 + similarity_boost

        print(f"Generating audio in language {language}...")
        output = self.model.synthesize(text,
            self.config,
            speaker_wav=speaker_audio_path,
            language=language,
            temperature=stability,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if os.path.exists("output.wav"):
            os.remove("output.wav")

        wavfile.write("output.wav", self.config.audio.sample_rate, output['wav'])
        
        return sieve.Audio(path="output.wav")
    




