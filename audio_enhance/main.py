import sieve
from deepfilternet import DeepFilterNetV2
from hifigan import HiFiGanPlus

wf_metadata = sieve.Metadata(
    title="Audio Enhance",
    description="Remove background noise from audio and upsample to 48kHz.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhance/main.py",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/cover.png"
    ),
    tags=["Audio"],
    readme=open("README.md", "r").read(),
)

@sieve.workflow(name="audio_background_noise_removal", metadata=wf_metadata)
def audio_enhance(audio: sieve.Audio) -> sieve.Audio:
    """
    :param audio: A noisy audio input (mp3 and wav supported)
    :return: Denoised audio
    """
    background_noise_removed =  DeepFilterNetV2()(audio)
    upsampled = HiFiGanPlus()(background_noise_removed)
    return upsampled

if __name__ == "__main__":
    sieve.push(
        workflow="audio_noise_reduction",
        inputs={
            "audio": {
                "url": "https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/input.wav"
            }
        },
    )
