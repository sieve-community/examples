# TODO: Make this work

import sieve

metadata = sieve.Metadata(
    title="Audio Enhancer",
    description="Remove background noise from audio and upsample to 48kHz.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/cover.png"
    ),
    tags=["Audio", "Speech", "Enhancement", "Featured"],
    readme=open("README.md", "r").read(),
)

@sieve.function(name="audio_enhancement", metadata=metadata)
def audio_enhance(audio: sieve.Audio) -> sieve.Audio:
    """
    :param audio: An audio input (mp3 and wav supported)
    :return: Enhanced audio
    """
    deepfilternet = sieve.function.get("sieve/deepfilternet_v2")
    upsampler = sieve.function.get("sieve/hifi_gan_plus")

    background_noise_removed = deepfilternet.run(audio)
    upsampled = upsampler.run(background_noise_removed)
    return upsampled
