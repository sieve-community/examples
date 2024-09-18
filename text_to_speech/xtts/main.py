import sieve
from typing import Literal

metadata = sieve.Metadata(
    description="Generate audio that sounds similar to a reference speaker in 13 languages",
    tags=["Audio", "Speech", "TTS"],
    image=sieve.Image(
        url="https://miro.medium.com/v2/resize:fit:512/0*bWnCJKvDb1wmPiPP"
    ),
    readme=open("README.md", "r").read(),
)

name_mapping = {
    "Claribel": "Claribel Dervla",
    "Daisy": "Daisy Studious",
    "Gracie": "Gracie Wise",
    "Tammie": "Tammie Ema",
    "Alison": "Alison Dietlinde",
    "Ana": "Ana Florence",
    "Annmarie": "Annmarie Nele",
    "Asya": "Asya Anara",
    "Brenda": "Brenda Stern",
    "Gitta": "Gitta Nikolina",
    "Henriette": "Henriette Usha",
    "Sofia": "Sofia Hellen",
    "Tammy": "Tammy Grit",
    "Tanja": "Tanja Adelina",
    "Vjollca": "Vjollca Johnnie",
    "Andrew": "Andrew Chipper",
    "Badr": "Badr Odhiambo",
    "Dionisio": "Dionisio Schuyler",
    "Royston": "Royston Min",
    "Viktor": "Viktor Eka",
    "Abrahan": "Abrahan Mack",
    "Adde": "Adde Michal",
    "Baldur": "Baldur Sanjin",
    "Craig": "Craig Gutsy",
    "Damien": "Damien Black",
    "Gilberto": "Gilberto Mathias",
    "Ilkin": "Ilkin Urbano",
    "Kazuhiko": "Kazuhiko Atallah",
    "Ludvig": "Ludvig Milivoj",
    "Suad": "Suad Qasim",
    "Torcull": "Torcull Diarmuid",
    "Zacharie": "Zacharie Aimilios",
    "Nova": "Nova Hogarth",
    "Maja": "Maja Ruoho",
    "Uta": "Uta Obando",
    "Lidiya": "Lidiya Szekeres",
    "Chandra": "Chandra MacFarland",
    "Szofi": "Szofi Granger",
    "Camilla": "Camilla Holmström",
    "Lilya": "Lilya Stainthorpe",
    "Zofija": "Zofija Kendrick",
    "Narelle": "Narelle Moon",
    "Barbora": "Barbora MacLean",
    "Alexandra": "Alexandra Hisakawa",
    "Alma": "Alma María",
    "Rosemary": "Rosemary Okafor",
    "Ige": "Ige Behringer",
    "Filip": "Filip Traverse",
    "Damjan": "Damjan Chapman",
    "Wulf": "Wulf Carlevaro",
    "Aaron": "Aaron Dreschner",
    "Kumar": "Kumar Dahl",
    "Eugenio": "Eugenio Mataracı",
    "Ferran": "Ferran Simen",
    "Xavier": "Xavier Hayasaka",
    "Luis": "Luis Moray",
    "Marcos": "Marcos Rudaski"
}


@sieve.Model(
    name="xtts",
    metadata=metadata,
    gpu=sieve.gpu.L4(),
    python_packages=[
        "TTS",
        "transformers",
    ],
    run_commands=[
        "mkdir -p /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/hash.md5 -q -P /root/.cache/model",
        "wget https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth -q -P /root/.cache/model",
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
        import os

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
            reference_audio: sieve.File,
            speaker: Literal["None", "Claribel", "Daisy", "Gracie", "Tammie", "Alison", "Ana", "Annmarie", "Asya", "Brenda", "Gitta", "Henriette", "Sofia", "Tammy", "Tanja", "Vjollca", "Andrew", "Badr", "Dionisio", "Royston", "Viktor", "Abrahan", "Adde", "Baldur", "Craig", "Damien", "Gilberto", "Ilkin", "Kazuhiko", "Ludvig", "Suad", "Torcull", "Zacharie", "Nova", "Maja", "Uta", "Lidiya", "Chandra", "Szofi", "Camilla", "Lilya", "Zofija", "Narelle", "Barbora", "Alexandra", "Alma", "Rosemary", "Ige", "Filip", "Damjan", "Wulf", "Aaron", "Kumar", "Eugenio", "Ferran", "Xavier", "Luis", "Marcos"] = "None",
            stability: float = 0.5,
            similarity_boost: float = 0.63,
            language_code: str = "",
        ) -> sieve.File:
        """
        :param text: text to speak
        :param reference_audio: audio of the reference speaker.
        :param speaker: pretrained speakers to use. reference_audio will be ignored if speaker is not "None".
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


        if speaker == "None":
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

        if speaker == "None":
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
        else:
            print(f"Generating audio in language {language} with speaker {speaker}...")
            output = self.model.synthesize(text,
                self.config,
                speaker_wav='demo.wav',
                language=language,
                speaker_id=name_mapping[speaker],
                temperature=stability,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        if os.path.exists("output.wav"):
            os.remove("output.wav")

        wavfile.write("output.wav", self.config.audio.sample_rate, output['wav'])
        
        return sieve.Audio(path="output.wav")



