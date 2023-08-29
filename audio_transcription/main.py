import sieve


@sieve.Model(
    name="whisperx",
    gpu=True,
    python_packages=[
        "torch==2.0",
        "torchaudio==2.0.0",
        "git+https://github.com/m-bain/whisperx.git@befe2b242eb59dcd7a8a122d127614d5c63d36e9",
    ],
    cuda_version="11.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    machine_type="T4-highmem-8-ssd",
    run_commands=[
        "mkdir -p /root/.cache/models/",
        "wget -c 'https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin' -P /root/.cache/models/",
        'python -c \'from faster_whisper.utils import download_model; download_model("large-v2", cache_dir="/root/.cache/models/")\'',
        "mkdir -p /root/.cache/torch/",
        "mkdir -p /root/.cache/torch/hub/",
        "mkdir -p /root/.cache/torch/hub/checkpoints/",
        "wget -c 'https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth' -P /root/.cache/torch/hub/checkpoints/",
    ],
)
class Whisper:
    def __setup__(self):
        import time
        import numpy as np
        import whisperx
        from whisperx.audio import load_audio
        from whisperx.asr import load_model

        self.model = load_model(
            "large-v2",
            "cuda",
            vad_options={"model_fp": "/root/.cache/models/pytorch_model.bin"},
            compute_type="int8",
            download_root="/root/.cache/models/",
        )
        # Pass in a dummy audio to warm up the model
        audio_np = np.zeros((32000 * 30), dtype=np.float32)
        self.model.transcribe(audio_np, batch_size=4)

        # Dummy load audio to warm up ffmpeg
        load_audio("silent_second.wav")

        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda"
        )

    def __predict__(self, audio: sieve.Audio) -> dict:
        """
        :param audio: Audio to transcribe
        :return: Dict with text, start, and end timestamps for each segment
        """

        import time
        import whisperx
        import numpy as np
        from whisperx.audio import load_audio

        overall_time = time.time()
        start_time = 0
        audio_np = load_audio(audio.path).astype(np.float32)
        # Pad to 30 seconds with silence
        if audio_np.shape[0] < 32000 * 30:
            audio_np = np.pad(audio_np, (0, 32000 * 30 - audio_np.shape[0]), "constant")

        result = self.model.transcribe(audio_np, batch_size=16)

        result_aligned = whisperx.align(
            result["segments"], self.model_a, self.metadata, audio_np, "cuda"
        )
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
                new_word["word"] = word["word"]
                new_segment["words"].append(new_word)

            out_segments.append(new_segment)
        print("overall_time: ", time.time() - overall_time)
        return out_segments


@sieve.workflow(name="audio_transcription")
def whisper_wf(audio: sieve.Audio) -> dict:
    """
    :param audio: Audio to transcribe
    :return: Dict with text, start, and end timestamps for each segment
    """
    whisper = Whisper()
    return whisper(audio)


if __name__ == "__main__":
    sieve.push(
        workflow="audio_transcription",
        inputs={
            "audio": {
                "url": "https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/input.wav"
            }
        },
    )
