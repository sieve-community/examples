import sieve

metadata = sieve.Metadata(
    description="Fast, high quality speech transcription with word-level timestamps and translation capabilities",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription",
    image=sieve.Image(
        url="https://i0.wp.com/opusresearch.net/wordpress/wp-content/uploads/2022/10/DALL%C2%B7E-2022-10-06-14.02.14-multilayered-beautiful-4k-hyper-realistic-neon-audio-waves-in-rainbow-colors-on-a-black-background.png?ssl=1"
    ),
    tags=["Audio", "Speech", "Transcription", "Featured"],
    readme=open("README.md", "r").read(),
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

@sieve.function(
    name="speech_transcriber",
    python_packages=[
        "librosa==0.8.0",
        "soundfile==0.12.1",
        "ffmpeg-python==0.2.0",
        "numpy==1.19.4",
    ],
    system_packages=["ffmpeg"],
    metadata=metadata,
)
def audio_split_by_silence(
    file: sieve.Audio,
    word_level_timestamps: bool = True,
    source_language: str = "",
    target_language: str = "",
    min_silence_length: float = 0.8,
    min_segment_length: float = 60.0
):
    '''
    :param file: Audio file
    :param word_level_timestamps: Whether to return word-level timestamps. Defaults to True.
    :param source_language: Language of the audio. Defaults to auto-detect if not specified. Otherwise, specify the language code {en, fr, de, es, it, ja, zh, nl, uk, pt}. This may improve transcription speed.
    :param target_language: Language code of the language to translate to (doesn't translate if left blank). See README for supported language codes.
    :param min_silence_length: Minimum length of silence in seconds to use for splitting audio for parallel processing. Defaults to 0.8.
    :param min_segment_length: Minimum length of audio segment in seconds to use for splitting audio for parallel processing. Defaults to 60.0.
    '''
    import os
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    import requests
    import subprocess

    # Extract the length of the audio using ffprobe
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file.path], capture_output=True, text=True)
    audio_length = float(result.stdout)

    min_silence_length = float(min_silence_length)
    min_segment_length = float(min_segment_length)
    import re

    def split_silences(
        path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
    ):
        import concurrent.futures
        import ffmpeg

        silence_end_re = re.compile(
            r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
        )

        metadata = ffmpeg.probe(path)
        duration = float(metadata["format"]["duration"])

        reader = (
            ffmpeg.input(str(path))
            .filter("silencedetect", n="-10dB", d=min_silence_length)
            .output("pipe:", format="null")
            .run_async(pipe_stderr=True)
        )

        cur_start = 0.0
        num_segments = 0
        count = 0

        while True:
            line = reader.stderr.readline().decode("utf-8")
            if not line:
                break
            match = silence_end_re.search(line)
            if match:
                silence_end, silence_dur = match.group("end"), match.group("dur")
                split_at = float(silence_end) - (float(silence_dur) / 2)

                if (split_at - cur_start) < min_segment_length:
                    continue

                yield cur_start, split_at
                cur_start = split_at
                num_segments += 1

        if duration > cur_start:
            yield cur_start, duration
            num_segments += 1
        print(f"Split {path} into {num_segments} segments")

    count = 0
    audio_path = file.path
    whisperx = sieve.function.get("sieve/whisperx")
    translate = sieve.function.get("sieve/seamless_text2text")

    # create a temporary directory to store the audio files
    import tempfile
    import os
    import shutil

    temp_dir = tempfile.mkdtemp()
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def process_segment(segment):
        start_time, end_time = segment
        print(f"Splitting audio from {start_time} to {end_time}")
        whisperx_job = whisperx.push(
            sieve.Audio(path=audio_path, start_time=start_time, end_time=end_time),
            language=source_language,
            word_level_timestamps=word_level_timestamps
        )
        return whisperx_job

    segments = split_silences(
        audio_path,
        min_silence_length=min_silence_length,
        min_segment_length=min_segment_length,
    )
    if not segments:
        segments.append(whisperx.push(sieve.Audio(path=audio_path)))

    for job in executor.map(process_segment, segments):
        job_output = job.result()
        job_segments = job_output["segments"]
        if len(job_segments) > 0:
            print(f"transcribed {100*job_segments[-1]['end'] / audio_length:.2f}% of {audio_length:.2f} seconds")
        if len(target_language) > 0 and job_output["language_code"] != target_language and job_output["text"]:
            if target_language not in whisper_to_seamless_languages:
                raise Exception(
                    f"Target language not supported for translation: ",
                    target_language,
                )
            if job_output["language_code"] not in whisper_to_seamless_languages:
                raise Exception(
                    f"Detected language not supported for translation: ",
                    job_output["language_code"],
                )

            # Output language is in Whisper's language coding, so we need to transform to seamless
            seamless_target_lang = whisper_to_seamless_languages[
                target_language
            ]
            seamless_source_lang = whisper_to_seamless_languages[
                job_output["language_code"]
            ]
            text = translate.run(
                job_output["text"],
                target_language=seamless_target_lang,
                source_language=seamless_source_lang,
            )
            modified_job_output = {}
            modified_job_output["text"] = job_output["text"]
            modified_job_output["language_code"] = job_output["language_code"]
            modified_job_output["translated_text"] = text
            modified_job_output["translated_language_code"] = target_language
            modified_job_output["segments"] = job_output["segments"]
            job_output = modified_job_output
        yield job_output
    
    print("transcription finished")
