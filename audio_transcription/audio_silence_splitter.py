import sieve

@sieve.function(
    name="audio_split_by_silence",
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
    ]
)
def audio_split_by_silence(audio: sieve.Audio) -> sieve.Audio:
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
                print('-------------')
                print(line)
                silence_end, silence_dur = match.group("end"), match.group("dur")
                split_at = float(silence_end) - (float(silence_dur) / 2)
                print(f"Splitting at {split_at}")
                print(f"Curr start {cur_start}")
                print(f"No. of segments {num_segments}")
                print(f"Is yield {split_at - cur_start >= min_segment_length}")

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
    audio_path = audio.path
    for start_time, end_time in split_silences(audio_path, min_silence_length=min_silence_length, min_segment_length=min_segment_length):
        print(f"Splitting {audio_path} from {start_time} to {end_time}")
        pth = str(count)
        count += 1
        yield sieve.Audio(path=audio_path, start_time=start_time, end_time=end_time)

    if count == 0:
        yield sieve.Audio(path=audio_path)
