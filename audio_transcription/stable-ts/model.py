import sieve
from typing import List
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

metadata = sieve.Metadata(
    description="An version of Whisper that offers accurate word-level timestamps and little hallucination.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription/stable-ts",
    image=sieve.Image(
        url="https://storage.googleapis.com/mango-public-models/stable-ts-icon.png"
    ),
    tags=["Audio", "Speech", "Transcription"],
    readme=open(os.path.join(file_dir, "README.md"), "r").read(),
)


@sieve.Model(
    name="stable-ts",
    gpu="l4",
    python_packages=[
        "git+https://github.com/jianfch/stable-ts.git",
    ],
    cuda_version="11.8",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "pip install faster-whisper",
        "python -c 'import stable_whisper as whisper; model = whisper.load_faster_whisper(\"large-v3\", device=\"cpu\")'",
        "python -c 'import stable_whisper as whisper; model = whisper.load_faster_whisper(\"base\", device=\"cpu\")'",
        "pip install python-dotenv",
    ],
    metadata=metadata
)
class Whisper:
    def __setup__(self):
        import time

        from dotenv import load_dotenv
        load_dotenv()

        start_time = time.time()
        import time
        t = time.time()
        import stable_whisper
        self.model = stable_whisper.load_faster_whisper('large-v3', device="cuda")
        self.model_medium = stable_whisper.load_faster_whisper('base', device="cuda")

        self.setup_time = time.time() - t
        self.first_time = True

        import numpy as np
        # Pass in a dummy audio to warm up the model
        audio_np = np.zeros((32000 * 30), dtype=np.float32)
        self.model.transcribe(audio_np)
        self.model_medium.transcribe(audio_np)


        self.diarize_model = sieve.function.get("sieve/pyannote-diarization")

        self.setup_time = time.time() - start_time
        self.first_time = True

    def load_audio(self, fp: str, start=None, end=None, sr: int = 16000):
        import ffmpeg
        import numpy as np

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
        language: str = "",
        diarize_min_speakers: int = -1,
        diarize_max_speakers: int = -1,
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

        if language == "":
            language = None

        if self.first_time:
            print("first_time_setup: ", self.setup_time)
            self.first_time = False
        import numpy as np

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
            audio_np = self.load_audio(audio_path)
            # if audio_np.shape[0] < 32000 * 30:
            #     audio_np = np.pad(
            #         audio_np, (0, 32000 * 30 - audio_np.shape[0]), "constant"
            #     )
        print("load_time: ", time.time() - t)

        process_time = time.time()
        if speed_boost:
            result = self.model_medium.transcribe_stable(audio_np, language=language, initial_prompt=initial_prompt, input_sr=16000, word_timestamps=word_level_timestamps)
        else:
            result = self.model.transcribe_stable(audio_np, language=language, initial_prompt=initial_prompt, input_sr=16000, word_timestamps=word_level_timestamps)
        
        result = result.to_dict()
        print("transcribe_time: ", time.time() - process_time)
        process_time = time.time()

        print("result: ", result)
        process_time = time.time()

        out_segments = []
        full_text = ""

        for segment in result["segments"]:
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
                    if "probability" in word:
                        new_word["confidence"] = word["probability"]
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
        return temp_out
        

if __name__=="__main__":
    import time
    start_time = time.time()
    model = Whisper()
    print("setup_time: ", time.time() - start_time)
    start_time = time.time()
    result = model(
        audio=sieve.File(
            path="/home/ubuntu/experiments/assets/downsampled.mp3"
        ),
        word_level_timestamps=False,
    )
    print("predict_time: ", time.time() - start_time)
    # print(result)