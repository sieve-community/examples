import sieve
from functools import lru_cache
import sieve
import soundfile as sf
import warnings
import os
import subprocess

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
        :param language: Language code of the audio (defaults to English), faster inference if the language is known. Leave blank for auto-detection.
        :return: a list of segments, each with a start time, end time, and text
        """
        from mosestokenizer import MosesTokenizer

        processor = OnlineASRProcessor(self.model, MosesTokenizer(language))

        if not os.path.exists("out"):
            os.makedirs("out")

        try:
            command = [
                'ffmpeg', '-i', url, '-f', 'segment', '-segment_time', '3', 
                '-reset_timestamps', '1', '-c:a', 'aac', '-vn', 'out/output_%03d.m4a'
            ]
            print(command)
            process = subprocess.Popen(
                command,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}") from e

        segment_re = re.compile(
            r"\[segment @ 0x[0-9a-f]+\] Opening '(out\/output_\d+.m4a)' for writing"
        )

        curr_chunk = 0
        prev_lines = []
        while True:
            try:
                line = process.stderr.readline()
                prev_lines.append(line)
                if len(prev_lines) > 10:
                    prev_lines.pop(0)
                if not line:
                    print('\n'.join(prev_lines))
                    break
                match = segment_re.search(line)
                if match:
                    if curr_chunk > 0:
                        audio_path = f"out/output_{(curr_chunk - 1):03d}.m4a"
                        a = load_audio_chunk(audio_path)
                        processor.insert_audio_chunk(a)

                        os.remove(audio_path)
                        try:
                            print("Processing chunk...")
                            o = processor.process_iter()
                            if o is not None and o[0] is not None:
                                print(f"yielding chunk from {o[0]} to {o[1]} with text: {o[2]}")
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
