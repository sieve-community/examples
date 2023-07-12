import sieve
from main import TortoiseTTS

@sieve.workflow(name="text-to-speech")
def tortoise_tts(text: str) -> sieve.Audio:
    return TortoiseTTS()(text)

