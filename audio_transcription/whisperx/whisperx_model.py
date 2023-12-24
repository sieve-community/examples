import sieve
from typing import List
from pydantic import BaseModel
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

metadata = sieve.Metadata(
    description="WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization)",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/whisperx",
    image=sieve.Image(
        url="https://github.com/m-bain/whisperX/raw/main/figures/pipeline.png"
    ),
    tags=["Audio", "Speech", "Transcription"],
    readme=open(os.path.join(file_dir, "WHISPERX_README.md"), "r").read(),
)


class Word(BaseModel):
    start: float
    end: float
    score: float
    word: str


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: List[Word]

@sieve.Model(
    name="whisperx",
    gpu="l4",
    python_packages=[
        "torch==2.0",
        "torchaudio==2.0.0",
        "git+https://github.com/m-bain/whisperx.git@e9c507ce5dea0f93318746411c03fed0926b70be",
    ],
    cuda_version="11.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models/",
        "wget -c 'https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin' -P /root/.cache/models/",
        'python -c \'from faster_whisper.utils import download_model; download_model("large-v3", cache_dir="/root/.cache/models/")\'',
        'python -c \'from faster_whisper.utils import download_model; download_model("base", cache_dir="/root/.cache/models/")\'',
        "mkdir -p /root/.cache/torch/",
        "mkdir -p /root/.cache/torch/hub/",
        "mkdir -p /root/.cache/torch/hub/checkpoints/",
        "wget -c 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth' -P /root/.cache/torch/hub/checkpoints/",
        "pip install ffmpeg-python",
        "pip install python-dotenv",
    ],
    metadata=metadata,
)
class Whisper:
    def __setup__(self):
        import os
        import time

        from dotenv import load_dotenv
        load_dotenv()

        start_time = time.time()
        import numpy as np
        import whisperx
        from whisperx.asr import load_model

        self.model = load_model(
            "large-v3",
            "cuda",
            # language="en",
            asr_options={
                "initial_prompt": os.getenv("initial_prompt"),
            },
            vad_options={"model_fp": "/root/.cache/models/pytorch_model.bin"},
            compute_type="int8",
            download_root="/root/.cache/models/",
        )

        self.model_medium = load_model(
            "base",
            "cuda",
            # language="en",
            asr_options={
                "initial_prompt": os.getenv("initial_prompt"),
            },
            vad_options={"model_fp": "/root/.cache/models/pytorch_model.bin"},
            compute_type="int8"
        )
        # Pass in a dummy audio to warm up the model
        audio_np = np.zeros((32000 * 30), dtype=np.float32)
        self.model.transcribe(audio_np, batch_size=4)

        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda"
        )

        self.diarize_model = sieve.function.get("sieve/pyannote-diarization")

        self.setup_time = time.time() - start_time
        self.first_time = True

    def load_audio(self, fp: str, start=None, end=None, sr: int = 16000):
        import ffmpeg
        import numpy as np
        import time

        try:
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
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def __predict__(
        self, audio: sieve.File,
        word_level_timestamps: bool = True,
        speaker_diarization: bool = False,
        speed_boost: bool = False,
        start_time: float = 0,
        end_time: float = -1,
        initial_prompt: str = "",
        prefix: str = "",
        language: str = "",
        diarize_min_speakers: int = -1,
        diarize_max_speakers: int = -1,
        batch_size: int = 32,
    ) -> List:
        """
        :param audio: an audio file
        :param word_level_timestamps: whether to return word-level timestamps
        :param speaker_diarization: whether to perform speaker diarization
        :param speed_boost: whether to use the smaller, faster model (large-v3 vs base)
        :param start_time: start time of the audio in seconds. Defaults to 0.
        :param end_time: end time of the audio in seconds. Defaults to -1 (end of audio).
        :param initial_prompt: A prompt to correct misspellings and style.
        :param prefix: A prefix to bias the transcript towards.
        :param language: Language code of the audio (defaults to English), faster inference if the language is known.
        :param diarize_min_speakers: Minimum number of speakers to detect. If set to -1, the number of speakers is automatically detected.
        :param diarize_max_speakers: Maximum number of speakers to detect. If set to -1, the number of speakers is automatically detected.
        :param batch_size: Batch size for inference. Defaults to 32.
        :return: a list of segments, each with a start time, end time, and text
        """
        import time
        overall_time = time.time()
        import faster_whisper

        new_asr_options = self.model.options._asdict()
        new_asr_options["initial_prompt"] = initial_prompt
        new_asr_options["prefix"] = prefix
        new_options = faster_whisper.transcribe.TranscriptionOptions(**new_asr_options)
        self.model.options = new_options

        if self.first_time:
            print("first_time_setup: ", self.setup_time)
            self.first_time = False
        import numpy as np
        from whisperx.audio import load_audio

        t = time.time()
        audio_path = audio.path

        if speaker_diarization:
            diarization_job = self.diarize_model.push(
                sieve.File(path=audio_path),
                min_speakers=diarize_min_speakers,
                max_speakers=diarize_max_speakers,
            )
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
        if speed_boost:
            result = self.model_medium.transcribe(audio_np, batch_size=batch_size, language=language)
        else:
            result = self.model.transcribe(audio_np, batch_size=batch_size, language=language)
        print("transcribe_time: ", time.time() - process_time)
        process_time = time.time()
        import whisperx

        if word_level_timestamps:
            result_aligned = whisperx.align(
                result["segments"], self.model_a, self.metadata, audio_np, "cuda"
            )

            print("align_time: ", time.time() - process_time)
        else:
            result_aligned = result
        process_time = time.time()

        out_segments = []
        full_text = ""
        for segment in result_aligned["segments"]:
            new_segment = {}
            new_segment["start"] = segment["start"] + start_time
            new_segment["end"] = segment["end"] + start_time
            new_segment["text"] = segment["text"]
            if "speaker" in segment:
                new_segment["speaker"] = segment["speaker"]
            
            full_text += segment["text"] + " "
            if word_level_timestamps:
                new_segment["words"] = []
                for word in segment["words"]:
                    new_word = {}
                    if "speaker" in word:
                        new_word["speaker"] = word["speaker"]
                    if "start" in word:
                        new_word["start"] = word["start"] + start_time
                    if "end" in word:
                        new_word["end"] = word["end"] + start_time
                    if "score" in word:
                        new_word["score"] = word["score"]
                    new_word["word"] = word["word"]
                    new_segment["words"].append(new_word)

            out_segments.append(new_segment)

        if speaker_diarization:
            diarization_job_output = diarization_job.result()
            print("diarization finished")
        
        temp_out = {
            "text": full_text.strip(),
            "language_code": result["language"],
            "segments": out_segments,
        }

        if speaker_diarization:
            transcript_segments = temp_out["segments"]
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
            print("overall_time: ", time.time() - overall_time)
        
        print("overall_time: ", time.time() - overall_time)
        return temp_out
        
