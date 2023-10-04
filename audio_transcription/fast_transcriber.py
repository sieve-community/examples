# TODO: make this work

import sieve

metadata = sieve.Metadata(
    description="High quality word-level speech transcription",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription",
    image=sieve.Image(
        url="https://i0.wp.com/opusresearch.net/wordpress/wp-content/uploads/2022/10/DALL%C2%B7E-2022-10-06-14.02.14-multilayered-beautiful-4k-hyper-realistic-neon-audio-waves-in-rainbow-colors-on-a-black-background.png?ssl=1"
    ),
    tags=["Audio", "Speech", "Transcription"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="speech_transcriber",
    python_packages=[
        "librosa==0.8.0",
        "soundfile==0.12.1",
        "ffmpeg-python==0.2.0",
        "numpy==1.19.4"
    ],
    system_packages=[
        "ffmpeg"
    ],
    environment_variables=[
        sieve.Env(name="min_silence_length", default= 0.8),
        sieve.Env(name="min_segment_length", default= 30.0)
    ],
    metadata=metadata
)
def audio_split_by_silence(file: sieve.Audio):
    import os
    import sys
    import librosa
    import numpy as np
    import soundfile as sf
    import requests
    min_silence_length = float(os.getenv("min_silence_length"))
    min_segment_length = float(os.getenv("min_segment_length"))
    from typing import Iterator
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
    transcription_jobs = []
    whisperx = sieve.function.get("sieve/whisperx")
    
    # create a temporary directory to store the audio files
    import tempfile
    import os
    import shutil
    temp_dir = tempfile.mkdtemp()
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def process_segment(segment):
        start_time, end_time = segment
        print(f"Splitting {audio_path} from {start_time} to {end_time}")
        whisperx_job = whisperx.push(sieve.Audio(path=audio_path, start_time=start_time, end_time=end_time))
        return whisperx_job

    segments = split_silences(audio_path, min_silence_length=min_silence_length, min_segment_length=min_segment_length)
    if not segments:
        transcription_jobs.append(whisperx.push(sieve.Audio(path=audio_path)))

    for job in executor.map(process_segment, segments):
        yield job.result()
