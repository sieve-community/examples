import sieve
from functools import lru_cache
import sieve
import ffmpeg
import soundfile as sf
import warnings
import os

warnings.filterwarnings("ignore")

import re
from utils import OnlineASRProcessor, load_audio_chunk


class SieveWhisper:
    sep = " "

    def __init__(self, lan="en", modelsize=None, cache_dir=None, model_dir=None):
        self.transcribe_kargs = {}
        self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        self.model = sieve.function.get("sieve/whisperx")
        return self.model

    def transcribe(self, audio, init_prompt=""):
        sf.write("./tmp.wav", audio, 16000)
        result = self.model.run(
            sieve.Audio(path="./tmp.wav"),
            initial_prompt=init_prompt,
            language=self.original_language,
        )
        return result

    def ts_words(self, r):
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                if "start" in w and "end" in w:
                    t = (w["start"], w["end"], w["word"])
                    o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]


@sieve.Model(
    name="live_speech_transcriber",
    python_version="3.11",
    python_packages=["librosa", "numpy", "ffmpeg-python", "opus-fast-mosestokenizer"],
    system_packages=["ffmpeg"],
)
class LiveSpeechTranscriber:
    def __setup__(self):
        self.model = SieveWhisper()

    def __predict__(self, url: str, language: str = "en"):
        """
        :param url: A URL to a live audio stream (RTMP, HLS, etc.). Needs to be supported by FFMPEG.
        :param language: Language code of the audio (defaults to English), faster inference if the language is known
        :return: a list of segments, each with a start time, end time, and text
        """
        from mosestokenizer import MosesTokenizer

        processor = OnlineASRProcessor(self.model, MosesTokenizer(language))
        reader = (
            ffmpeg.input(url)
            .output(
                "out/output_%03d.mpeg",
                format="segment",
                segment_time="3",
                reset_timestamps="1",
                c="copy",
            )
            .run_async(pipe_stderr=True)
        )

        segment_re = re.compile(
            r"\[segment @ 0x[0-9a-f]+\] Opening '(out\/output_\d+.mpeg)' for writing"
        )

        curr_chunk = 0
        while True:
            try:
                line = reader.stderr.readline().decode("utf-8")
                if not line:
                    break
                match = segment_re.search(line)
                if match:
                    if curr_chunk > 0:
                        audio_path = f"out/output_{(curr_chunk - 1):03d}.mpeg"
                        a = load_audio_chunk(audio_path)
                        processor.insert_audio_chunk(a)

                        # remove the audio chunk
                        os.remove(audio_path)
                        try:
                            print("Processing chunk: ", curr_chunk)
                            o = processor.process_iter()
                            if o is not None and o[0] is not None:
                                print(o[2], flush=True, end=" ")
                                yield {
                                    "start": o[0],
                                    "end": o[1],
                                    "text": o[2],
                                }
                        except AssertionError as e:
                            print("AssertionError:", e)
                            pass
                    curr_chunk += 1
            except KeyboardInterrupt:
                print("Closing ffmpeg...")
                reader.terminate()
                reader.wait()
                break
