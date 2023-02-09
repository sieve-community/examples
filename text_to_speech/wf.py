import sieve
from main import TortoiseTTS

@sieve.workflow(name="tortoise_tts")
def tortoise_tts(text: str) -> sieve.Audio:
    return TortoiseTTS()(text)

