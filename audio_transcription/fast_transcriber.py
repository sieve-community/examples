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
    file: sieve.File,
    word_level_timestamps: bool = True,
    speaker_diarization: bool = False,
    speed_boost: bool = False,
    source_language: str = "",
    target_language: str = "",
    min_speakers: int = -1,
    max_speakers: int = -1,
    min_silence_length: float = 0.8,
    min_segment_length: float = -1
):
    '''
    :param file: Audio file
    :param word_level_timestamps: Whether to return word-level timestamps. Defaults to True.
    :param speaker_diarization: Whether to perform speaker diarization. Defaults to False.
    :param speed_boost: Whether to use a faster, smaller transcription model. Defaults to False.
    :param source_language: Language of the audio. Defaults to auto-detect if not specified. Otherwise, specify the language code {en, fr, de, es, it, ja, zh, nl, uk, pt}. This may improve transcription speed.
    :param target_language: Language code of the language to translate to (doesn't translate if left blank). See README for supported language codes.
    :param min_speakers: Minimum number of speakers to detect for diarization. Defaults to auto-detect when set to -1.
    :param max_speakers: Maximum number of speakers to detect for diarization. Defaults to auto-detect when set to -1.
    :param min_silence_length: Minimum length of silence in seconds to use for splitting audio for parallel processing. Defaults to 0.8.
    :param min_segment_length: Minimum length of audio segment in seconds to use for splitting audio for parallel processing. Defaults to audio length / 20 if set to -1.
    '''
    import os
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    import requests
    import subprocess

    # Do diarization if specified
    if speaker_diarization:
        pyannote = sieve.function.get("sieve/pyannote-diarization")
        diarization_job = pyannote.push(
            sieve.File(path=file.path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        print("Warning: because speaker diarization is enabled, the transcription output will only return at the end of the job rather than when each segment is finished processing.")

    # Extract the length of the audio using ffprobe
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file.path], capture_output=True, text=True)
    audio_length = float(result.stdout)

    min_silence_length = float(min_silence_length)
    min_segment_length = float(min_segment_length)
    if min_segment_length < 0:
        min_segment_length = audio_length / 20
    import re

    def split_silences(
        path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
    ):
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
    import os

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

    def process_segment(segment):
        import time
        t = time.time()
        start_time, end_time = segment

        whisperx_job = whisperx.push(
            sieve.File(path=file.path),
            language=source_language,
            word_level_timestamps=word_level_timestamps,
            speed_boost=speed_boost,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"Took {time.time() - t:.2f} seconds to push segment from {start_time:.2f} to {end_time:.2f}")
        return whisperx_job

    segments = split_silences(
        audio_path,
        min_silence_length=min_silence_length,
        min_segment_length=min_segment_length,
    )
    if not segments:
        segments.append(whisperx.push(sieve.File(path=file.path)))

    job_outputs = []
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
        
        job_outputs.append(job_output)
        if not speaker_diarization:
            yield job_output
    
    if speaker_diarization:
        diarization_job_output = diarization_job.result()
        print("diarization finished")
        for job_output in job_outputs:
            transcript_segments = job_output["segments"]
            for seg in transcript_segments:
                diarization_segments = [s for s in diarization_job_output if s['end'] >= seg['start'] and s['start'] <= seg['end']]
                if diarization_segments:
                    seg["speaker"] = max(diarization_segments, key=lambda x: x['end'] - x['start'])['speaker_id']
                if 'words' in seg:
                    for word in seg['words']:
                        if 'start' in word:
                            diarization_segments = [s for s in diarization_job_output if s['end'] >= word['start'] and s['start'] <= word['end']]
                            if diarization_segments:
                                word["speaker"] = max(diarization_segments, key=lambda x: x['end'] - x['start'])['speaker_id']
            yield job_output
    
    print("transcription finished")
