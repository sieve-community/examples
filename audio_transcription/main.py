import sieve
from whisperx_model import Whisper
from audio_silence_splitter import audio_split_by_silence

wf_metadata = sieve.Metadata(
    title="Hyperfast Word Level Speech Transcription",
    description="Transcribe over 40 mins of audio in 30 seconds with word-level timestamps.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/main.py",
    image=sieve.Image(
        url="https://github.com/m-bain/whisperX/raw/main/figures/pipeline.png"
    ),
    tags=["Audio"],
    readme=open("README.md", "r").read(),
)

@sieve.workflow(name="whisperx_transcription", metadata=wf_metadata)
def whisper_wf(audio: sieve.Audio) -> dict:
    """
    :param audio: audio to transcribe
    :return: dictionary with text, start, and end timestamps for each word and segment
    """
    whisper = Whisper()
    return whisper(audio_split_by_silence(audio))


if __name__ == "__main__":
    sieve.push(
        workflow="audio_transcription",
        inputs={
            "audio": {
                "url": "https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/input.wav"
            }
        },
    )
