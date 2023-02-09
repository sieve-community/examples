import sieve 
from typing import Dict

@sieve.Model(
    name="whisper",
    gpu = True,
    python_packages=["git+https://github.com/openai/whisper.git"],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models",
        "wget -c 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt' -P /root/.cache/models"
    ]
)
class Whisper:
    def __setup__(self):
        import whisper
        self.model = whisper.load_model("/root/.cache/models/tiny.en.pt")

    def __predict__(self, audio: sieve.Audio) -> Dict:
        result = self.model.transcribe(audio.path)
        segments = result["segments"]
        for segment in segments:
            yield {
                'text': segment["text"],
                'start': segment["start"],
                'end': segment["end"]
            }

@sieve.workflow(name="whisper")
def whisper_wf(audio: sieve.Audio) -> sieve.Dict:
    whisper = Whisper()
    return whisper(audio)
