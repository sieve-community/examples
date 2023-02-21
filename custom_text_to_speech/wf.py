import sieve
from main import TortoiseTTS

@sieve.workflow(name="custom_tortoise_tts")
def tortoise_tts(text: str, preset: str, audio1: sieve.Audio, audio2: sieve.Audio) -> sieve.Audio:
    return TortoiseTTS()(text, preset, audio1, audio2)