import sieve
from typing import List
from pydantic import BaseModel
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

metadata = sieve.Metadata(
    description="Diarize audio using pyannote-audio",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/pyannote_diarization",
    image=sieve.Image(
        url="https://avatars.githubusercontent.com/u/7559051?s=280&v=4"
    ),
    tags=["Audio", "Speech"],
    readme=open(os.path.join(file_dir, "README.md"), "r").read(),
)

@sieve.Model(
    name="pyannote-diarization",
    gpu="a100",
    python_packages=[
        "torch==2.0",
        "torchaudio==2.0.0",
        "git+https://github.com/m-bain/whisperx.git@e9c507ce5dea0f93318746411c03fed0926b70be",
        "onnxruntime-gpu==1.16.0"
    ],
    cuda_version="11.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "pip install pyannote-audio==3.1.1",
        "pip uninstall onnxruntime -y",
        "pip install --force-reinstall onnxruntime-gpu==1.16.0",
        "pip install ffmpeg-python",
    ],
    metadata=metadata
)
class PyannoteDiarization:
    def __setup__(self):
        import time

        start_time = time.time()

        from pyannote.audio import Pipeline
        import torch
        self.diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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").to(torch.device("cuda"))

        self.setup_time = time.time() - start_time
        self.first_time = True

    def load_audio(self, fp: str, start=None, end=None, sr: int = 16000):
        import ffmpeg
        import numpy as np
        import time
        from whisperx.audio import load_audio

        try:
            start_time = time.time()
            if start is None and end is None:
                out, _ = (
                    ffmpeg.input(fp, threads=0)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                )
            else:
                out, _ = (
                    ffmpeg.input(fp, threads=0)
                    .filter("atrim", start=start, end=end)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                )
            end_time = time.time()
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def __predict__(
        self, audio: sieve.File,
        start_time: float = 0,
        end_time: float = -1,
        diarize_min_speakers: int = -1,
        diarize_max_speakers: int = -1,
    ) -> List:
        """
        :param audio: an audio file
        :param start_time: start time of the audio in seconds. Defaults to 0.
        :param end_time: end time of the audio in seconds. Defaults to -1 (end of audio).
        :param min_speakers: Minimum number of speakers to detect. If set to -1, the number of speakers is automatically detected.
        :param max_speakers: Maximum number of speakers to detect. If set to -1, the number of speakers is automatically detected.
        :param batch_size: Batch size for inference. Defaults to 32.
        """
        import time
        overall_time = time.time()

        if self.first_time:
            print("first_time_setup: ", self.setup_time)
            self.first_time = False
        import numpy as np
        from whisperx.audio import load_audio

        t = time.time()
        audio_path = audio.path
        print("get_audio_path_time: ", time.time() - t)
        if (hasattr(audio, "start_time") and hasattr(audio, "end_time")):
            import time

            t = time.time()
            start_time = audio.start_time
            end_time = audio.end_time
            audio_np = self.load_audio(audio_path, start=start_time, end=end_time)
        elif end_time != -1:
            import time
            t = time.time()
            audio_np = self.load_audio(audio_path, start=start_time, end=end_time)
        else:
            t = time.time()
            audio_np = load_audio(audio_path).astype(np.float32)
            if audio_np.shape[0] < 32000 * 30:
                audio_np = np.pad(
                    audio_np, (0, 32000 * 30 - audio_np.shape[0]), "constant"
                )
        print("load_time: ", time.time() - t)

        process_time = time.time()

        import torch
        from whisperx.audio import SAMPLE_RATE
        import pandas as pd

        min_speakers = diarize_min_speakers if diarize_min_speakers != -1 else None
        max_speakers = diarize_max_speakers if diarize_max_speakers != -1 else None
        audio_data = {
            'waveform': torch.from_numpy(audio_np[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        diarize_segments = self.diarize_model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        print("diarize_time: ", time.time() - process_time)

        print("diarize_df: ", diarize_df)

        def assign_word_speakers(diarize_df, fill_nearest=False):
            speaker_segments = []
            for _, seg in diarize_df.iterrows():
                # assign speaker to segment (if any)
                diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
                diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
                # remove no hit, otherwise we look for closest (even negative intersection...)
                if not fill_nearest:
                    dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                else:
                    dia_tmp = diarize_df
                if len(dia_tmp) > 0:
                    # sum over speakers
                    speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    seg["speaker"] = speaker
                else:
                    seg["speaker"] = "NO_SPEAKER"
                speaker_segments.append({
                    'speaker_id': seg['speaker'],
                    'start': seg['start'],
                    'end': seg['end']
                })
            return speaker_segments

        speaker_segments = assign_word_speakers(diarize_df)

        print("overall_time: ", time.time() - overall_time)
        return speaker_segments
