import sieve
from functools import lru_cache
import sieve
import soundfile as sf
import warnings
import os
import subprocess
import numpy as np
import time

warnings.filterwarnings("ignore")

metadata = sieve.Metadata(
    title="Transcribe a Live Stream",
    description="Transcribe and translate live audio streams in real-time with high accuracy.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/live_transcriber.py",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/live_transcriber.jpg"
    ),
    tags=["Audio", "Speech", "Transcription", "Showcase"],
    readme=open("LIVE_README.md", "r").read(),
)

whisper_to_seamless_languages = {
    "en": "eng",
    "zh": "cmn",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "ko": "kor",
    "fr": "fra",
    "ja": "jpn",
    "pt": "por",
    "tr": "tur",
    "pl": "pol",
    "ca": "cat",
    "nl": "nld",
    "ar": "arb",
    "sv": "swe",
    "it": "ita",
    "id": "ind",
    "hi": "hin",
    "fi": "fin",
    "vi": "vie",
    "he": "heb",
    "uk": "ukr",
    "el": "ell",
    "ms": "zsm",
    "cs": "ces",
    "ro": "ron",
    "da": "dan",
    "hu": "hun",
    "ta": "tam",
    "no": "nob",
    "th": "tha",
    "ur": "urd",
    "hr": "hrv",
    "bg": "bul",
    "lt": "lit",
    "cy": "cym",
    "sk": "slk",
    "te": "tel",
    "bn": "ben",
    "sr": "srp",
    "sl": "slv",
    "kn": "kan",
    "et": "est",
    "mk": "mkd",
    "eu": "eus",
    "is": "isl",
    "hy": "hye",
    "bs": "bos",
    "kk": "kaz",
    "gl": "glg",
    "mr": "mar",
    "pa": "pan",
    "km": "khm",
    "sn": "sna",
    "yo": "yor",
    "so": "som",
    "af": "afr",
    "ka": "kat",
    "be": "bel",
    "tg": "tgk",
    "sd": "snd",
    "gu": "guj",
    "am": "amh",
    "lo": "lao",
    "nn": "nno",
    "mt": "mlt",
    "my": "mya",
    "tl": "tgl",
    "as": "asm",
    "jw": "jav",
    "yue": "yue",
}


def split_last_silence(
    audio, threshold=-2500, min_silence_duration=0.8, sample_rate=16000
):
    """
    Returns a tuple with the first element being the audio before the last silence and the second element being the audio after the last silence
    """

    is_silence = audio < threshold
    indices = np.where(is_silence)[0]

    if not any(indices):
        # No silence detected, return the original audio
        print("No silences detected in chunk")
        return audio, np.array([])

    last_silence_start = None

    for idx in reversed(indices):
        if len(audio) - idx > int(min_silence_duration * sample_rate):
            # Found the last silence segment longer than min_silence_duration
            last_silence_start = idx
            break

    if last_silence_start is not None:
        # If the last silence segment is long enough, split the audio
        audio_before_silence = audio[:last_silence_start]
        audio_after_silence = audio[last_silence_start:]
        return audio_before_silence, audio_after_silence
    else:
        print("No silence long enough detected in chunk")
        # If no silence segment is long enough, return the original audio
        return audio, np.array([])


@sieve.function(
    name="live_speech_transcriber",
    python_packages=["librosa", "numpy", "ffmpeg-python", "soundfile"],
    system_packages=["ffmpeg"],
    metadata=metadata,
)
def live_transcriber(
    url: str,
    target_language: str = "en",
    stream_language: str = "",
    chunk_size: int = 5,
):
    """
    :param url: A URL to a live audio stream (RTMP, HLS, etc.). Needs to be supported by FFMPEG.
    :param target_language: Language code of the language of the transcript (defaults to English if blank). You can specify multiple languages by separating them with commas. See README for supported language codes.
    :param stream_language: Language code of the provided audio. Defaults to blank for auto-detection, but faster inference if the language is known.
    :param chunk_size: The interval at which to process transcripts. Must be > than 3 seconds.
    :return: a list of segments, each with a start time, end time, and text
    """
    if target_language.strip() == "":
        target_language = "eng"
    
    target_languages = [lang.strip() for lang in target_language.split(",")]

    if chunk_size < 3:
        chunk_size = 3

    transcribe = sieve.function.get("sieve/whisperx")
    translate = sieve.function.get("sieve/seamless_text2text")

    channels, sample_rate = 1, 16000
    audio_command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        url,
        "-map",
        "0:a:0",
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]

    process = subprocess.Popen(
        audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    print(" ".join(audio_command))

    samples_per_second = int(sample_rate * channels) * 2  # 1 second of audio

    curr_offset = 0
    curr_chunk = 0
    last_after_silence = None
    prev_transcript = "This is a transcript from a live stream, passed in chunks. "

    start_time = time.time()
    timeout_min = 1
    while True:
        # Note: Remove this check to run on a stream indefinitely.
        if time.time() - start_time > timeout_min * 60:
            print(
                f"This demo times out after {timeout_min} minute(s). Please refer to https://docs.sievedata.com/guide/examples/live-audio-transcription to deploy on your own."
            )
            break

        try:
            audio_data = process.stdout.read(samples_per_second * chunk_size)
            if not audio_data:
                print("No more audio data to read")
                break

            audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
            before_silence, after_silence = split_last_silence(audio_chunk)
            if last_after_silence is not None:
                before_silence = np.concatenate((last_after_silence, before_silence))

            sf.write("./temp.wav", before_silence, sample_rate)
            last_after_silence = after_silence.copy()

            secs = before_silence.shape[0] / sample_rate
            curr_chunk += 1
            try:
                print("Processing chunk...")
                prev_transcript = " ".join(prev_transcript.split(" ")[-200:])

                o = transcribe.run(
                    sieve.File(path="./temp.wav"),
                    initial_prompt=prev_transcript,
                    language=stream_language,
                )

                if o is not None and o["text"] is not None:
                    # If the previous text ends with punctuation, let's capitalize. If not, let's lowercase.
                    if prev_transcript.rstrip()[-1] in [".", "!", "?"]:
                        o["text"] = o["text"].capitalize()
                    else:
                        o["text"] = o["text"].lower()

                    # If the text ends in ..., remove it
                    if o["text"].rstrip()[-3:] == "...":
                        o["text"] = o["text"][:-3]

                    text = o["text"]
                    output_texts = {}
                    import concurrent.futures

                    def translate_text(target_language):
                        if o["language_code"] != target_language and o["text"]:
                            if target_language not in whisper_to_seamless_languages:
                                raise Exception(
                                    f"Target language not supported for translation: ",
                                    target_language,
                                )
                            if o["language_code"] not in whisper_to_seamless_languages:
                                raise Exception(
                                    f"Whisper detected language not supported for translation: ",
                                    o["language_code"],
                                )

                            # Output language is in Whisper's language coding, so we need to transform to seamless
                            seamless_target_lang = whisper_to_seamless_languages[
                                target_language
                            ]
                            seamless_source_lang = whisper_to_seamless_languages[
                                o["language_code"]
                            ]
                            return target_language, translate.run(
                                text,
                                target_language=seamless_target_lang,
                                source_language=seamless_source_lang,
                            )
                        return target_language, None

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_language = {executor.submit(translate_text, target_language): target_language for target_language in target_languages}
                        for future in concurrent.futures.as_completed(future_to_language):
                            target_language, translation = future.result()
                            if translation:
                                output_texts[target_language] = translation

                    data = {
                        "texts": output_texts,
                        "start": curr_offset,
                        "end": curr_offset + secs,
                        "original_text": o["text"],
                    }

                    yield data

                    curr_offset += secs
                    prev_transcript += o["text"] + " "

            except AssertionError as e:
                print("AssertionError:", e)
                pass
        except KeyboardInterrupt:
            print("Closing ffmpeg...")
            process.terminate()
            break


if __name__ == "__main__":
    start_time = time.time()
    url = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"

    vals = live_transcriber(url, target_language="es")
    transcript = ""
    for val in vals:
        transcript += val["text"] + " "

    print()
    print(transcript)
    print("Time taken: ", time.time() - start_time)
