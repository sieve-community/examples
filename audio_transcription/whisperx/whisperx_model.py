import sieve

metadata = sieve.Metadata(
    description="WhisperX: Automatic Speech Recognition with Word-level Timestamps (& Diarization)",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/whisperx",
    image=sieve.Image(
        url="https://github.com/m-bain/whisperX/raw/main/figures/pipeline.png"
    ),
    tags=["Audio", "Speech", "Transcription"],
    readme=open("WHISPERX_README.md", "r").read(),
)

@sieve.Model(
    name="whisperx",
    gpu = True,
    python_packages=[
        "torch==2.0",
        "torchaudio==2.0.0",
        "git+https://github.com/m-bain/whisperx.git@07fafa37b3ef7ce8628b194da302a5a996bb7d37"
    ],
    cuda_version="11.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models/",
        "wget -c 'https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin' -P /root/.cache/models/",
        "python -c 'from faster_whisper.utils import download_model; download_model(\"large-v2\", cache_dir=\"/root/.cache/models/\")'",
        "mkdir -p /root/.cache/torch/",
        "mkdir -p /root/.cache/torch/hub/",
        "mkdir -p /root/.cache/torch/hub/checkpoints/",
        "wget -c 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth' -P /root/.cache/torch/hub/checkpoints/",
    ],
    metadata=metadata
)
class Whisper:
    def __setup__(self):
        import os
        import time
        start_time = time.time()
        import numpy as np
        import whisperx
        from whisperx.audio import load_audio
        from whisperx.asr import load_model
        self.model = load_model(
            "large-v2",
            "cuda",
            # language="en",
            asr_options={
                "initial_prompt": os.getenv("initial_prompt"),
            },
            vad_options={
                "model_fp": "/root/.cache/models/pytorch_model.bin"
            },
            compute_type="int8",
            download_root="/root/.cache/models/"
        )
        # Pass in a dummy audio to warm up the model
        audio_np = np.zeros((32000 * 30), dtype=np.float32)
        self.model.transcribe(audio_np, batch_size=4)

        self.model_a, self.metadata = whisperx.load_align_model(language_code="en", device="cuda")

        self.setup_time = time.time() - start_time
        self.first_time = True
        # pass


    def load_audio(self, fp: str, start=None, end=None, sr: int = 16000):
        import ffmpeg
        import numpy as np
        import time
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

    def __predict__(self, audio: sieve.Audio) -> sieve.Audio:
        import time
        if self.first_time:
            print("first_time_setup: ", self.setup_time)
            self.first_time = False
        import numpy as np
        from whisperx.audio import load_audio
        overall_time = time.time()
        start_time = 0
        if hasattr(audio, "start_time") and hasattr(audio, "end_time"):
            import time
            t = time.time()
            start_time = audio.start_time
            end_time = audio.end_time
            audio_np = self.load_audio(audio.path, start=start_time, end=end_time)
        else:
            t = time.time()
            audio_np = load_audio(audio.path).astype(np.float32)
            if audio_np.shape[0] < 32000 * 30:
                audio_np = np.pad(audio_np, (0, 32000 * 30 - audio_np.shape[0]), "constant")

        result = self.model.transcribe(audio_np, batch_size=16)
        import whisperx
        result_aligned = whisperx.align(result["segments"], self.model_a, self.metadata, audio_np, "cuda")
        out_segments = []
        for segment in result_aligned["segments"]:
            new_segment = {}
            new_segment["start"] = segment["start"] + start_time
            new_segment["end"] = segment["end"] + start_time
            new_segment["text"] = segment["text"]
            new_segment["words"] = []
            for word in segment["words"]:
                new_word = {}
                if "start" in word:
                    new_word["start"] = word["start"] + start_time
                if "end" in word:
                    new_word["end"] = word["end"] + start_time
                if "score" in word:
                    new_word["score"] = word["score"]
                new_word["word"] = word["word"]
                new_segment["words"].append(new_word)

            out_segments.append(new_segment)
        print("overall_time: ", time.time() - overall_time)
        return out_segments
