import sieve
from language_maps import *

metadata = sieve.Metadata(
    title="Transcribe Speech",
    description="Fast, high quality speech transcription with word-level timestamps and translation capabilities",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_transcription",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/transcribe.jpg"
    ),
    tags=["Audio", "Speech", "Transcription", "Featured"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="speech_transcriber",
    python_packages=[
        "librosa==0.8.0",
        "soundfile==0.12.1",
        "ffmpeg-python==0.2.0",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "onnxruntime==1.16.3",
    ],
    system_packages=["libsox-dev", "sox", "ffmpeg"],
    run_commands=[
        "python -c 'import torch; torch.hub.load(repo_or_dir=\"snakers4/silero-vad\", model=\"silero_vad\", onnx=True)'"
    ],
    metadata=metadata
)
class SpeechTranscriber:
    def __setup__(self):
        import torch
        try:
            print("Loading model")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=True,
            )
            self.model = model
            self.utils = utils
            print("Model loaded")
        except Exception as e:
            self.model = None
            self.utils = None
    
    def split_silences_by_ffmpeg_detect(
        self, path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8
    ):
        import re
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
    
    def split_silences_by_silero_vad(
        self, path: str, min_segment_length: float = 30.0, min_silence_length: float = 0.8, vad_threshold: float = 0.2
    ):
        import ffmpeg
        # get audio duration and sample rate
        metadata = ffmpeg.probe(path)
        duration = float(metadata["format"]["duration"])
        audio_sample_rate = int(metadata["streams"][0]["sample_rate"])
        SAMPLING_RATE = 16000

        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = (
            self.utils
        )
        wav = read_audio(path, sampling_rate=audio_sample_rate)
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(
            wav, self.model, sampling_rate=SAMPLING_RATE, return_seconds=True, min_silence_duration_ms=min_silence_length * 1000, threshold=vad_threshold
        )

        
        num_segments = 0

        if len(speech_timestamps) == 0:
            cur_start = 0.0
        else:
            cur_start = speech_timestamps[0]["start"]
            cur_start = max(cur_start - 1, 0.0)

        for item in speech_timestamps:
            start, end = item["start"], item["end"]
            # convert start and end to correct sample rate
            start = start * (SAMPLING_RATE / audio_sample_rate)
            end = end * (SAMPLING_RATE / audio_sample_rate)
            if (end - cur_start) < min_segment_length:
                continue
            yield cur_start, end
            cur_start = end
            num_segments += 1

        if duration > cur_start and len(speech_timestamps) > 0:
            yield cur_start, duration
            num_segments += 1
        print(f"Split {path} into {num_segments} segments")
    
    def split_silences_by_pyannote(
        self, path: str, diarization_job_output: list, min_segment_length: float = 30.0
    ):
        import ffmpeg
        # get audio duration and sample rate
        metadata = ffmpeg.probe(path)
        duration = float(metadata["format"]["duration"])

        segments = [(seg["start"], seg["end"]) for seg in diarization_job_output]
        # they may overlap, so we need to merge them
        new_segments = []
        for seg in segments:
            if not new_segments:
                new_segments.append(seg)
            else:
                last_seg = new_segments[-1]
                if seg[0] - last_seg[1] < 0.5:
                    new_segments[-1] = (last_seg[0], seg[1])
                else:
                    new_segments.append(seg)
        
        num_segments = 0
        if len(new_segments) == 0:
            cur_start = 0.0
        else:
            cur_start = new_segments[0][0]
            cur_start = max(cur_start - 1, 0.0)

        for seg in new_segments:
            start, end = seg
            if (end - cur_start) < min_segment_length:
                continue
            yield cur_start, end
            cur_start = end
            num_segments += 1

        if len(new_segments) > 0 and new_segments[-1][1] > cur_start:
            yield cur_start, new_segments[-1][1]
            num_segments += 1
        
        print(f"Split into {num_segments} segments")
        

    def __predict__(
        self,
        file: sieve.File,
        word_level_timestamps: bool = True,
        speaker_diarization: bool = False,
        speed_boost: bool = False,
        backend: str = "stable-ts",
        source_language: str = "",
        target_language: str = "",
        min_speakers: int = -1,
        max_speakers: int = -1,
        min_silence_length: float = 0.4,
        min_segment_length: float = -1,
        chunks: str = "",
        denoise_audio: bool = False,
        use_vad: bool = False,
        use_pyannote_segmentation: bool = False,
        vad_threshold: float = 0.2,
        initial_prompt: str = "",
    ):
        '''
        :param file: Audio file
        :param word_level_timestamps: Whether to return word-level timestamps. Defaults to True.
        :param speaker_diarization: Whether to perform speaker diarization. Defaults to False.
        :param speed_boost: Whether to use a smaller, less accurate model for faster speed. Defaults to False.
        :param backend: A choice between different model backends. Choices between "stable-ts", "whisper-timestamped", and "whisperx". See README for more information.
        :param source_language: Language of the audio. Defaults to auto-detect if not specified. See README for supported language codes.
        :param target_language: Language code of the language to translate to (doesn't translate if left blank). See README for supported language codes.
        :param min_speakers: Minimum number of speakers to detect for diarization. Defaults to auto-detect when set to -1.
        :param max_speakers: Maximum number of speakers to detect for diarization. Defaults to auto-detect when set to -1.
        :param min_silence_length: Minimum length of silence in seconds to use for splitting audio for parallel processing. Defaults to 0.4.
        :param min_segment_length: Minimum length of audio segment in seconds to use for splitting audio for parallel processing. If set to -1, we pick a value based on your settings.
        :param chunks: A parameter to manually specify the start and end times of each chunk when splitting audio for parallel processing. If set to "", we use silence detection to split the audio. If set to a string formatted with a start and end second on each line, we use the specified chunks. Example: '0,10' and '10,20' on separate lines.
        :param denoise_audio: Whether to apply denoising to the audio to get rid of background noise before transcription. Defaults to False.
        :param use_vad: Whether to use Silero VAD for splitting audio into segments. Defaults to False. More accurate than ffmpeg silence detection.
        :param use_pyannote_segmentation: Whether to use Pyannote segmentation for splitting audio into segments. Defaults to False.
        :param vad_threshold: The threshold for VAD. Defaults to 0.2.
        :param initial_prompt: A prompt to correct misspellings and style. Defaults to "".
        '''
        import time
        print("Starting transcription...")
        t = time.time()
        file_path = file.path
        download_time = time.time() - t
        print(f"Downloaded file in {download_time:.2f} seconds")
        import subprocess

        if use_vad and use_pyannote_segmentation:
            raise ValueError("Cannot use both VAD and Pyannote segmentation at the same time. Please choose one.")
        
        if download_time > 5:
            # cache the file
            file = sieve.File(path=file_path)

        if use_vad:
            # create wav file
            new_audio_path = "file.wav"
            subprocess.run(["ffmpeg", "-y", "-i", file.path, new_audio_path])
            file = sieve.File(path=new_audio_path)

        # denoise the audio if specified
        if denoise_audio:
            print("Denoising audio...")
            denoiser = sieve.function.get("sieve/resemble-enhance")
            denoised_file = denoiser.run(sieve.Audio(path=file.path), process="denoise")
            file = denoised_file
            print("Denoising complete.")

        if source_language == "auto":
            source_language = ""

        # Do diarization if specified
        if speaker_diarization or use_pyannote_segmentation:
            print("Pushing speaker diarization job...")
            pyannote = sieve.function.get("sieve/pyannote-diarization")
            diarization_job = pyannote.push(
                sieve.File(path=file.path),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            if speaker_diarization:
                print("Warning: because speaker diarization is enabled, the transcription output will only return at the end of the job rather than when each segment is finished processing.")

        # Extract the length of the audio using ffprobe
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", f"{file.path}"], capture_output=True, text=True)
        try:
            audio_length = float(result.stdout.strip())
        except:
            try:
                # try using librosa to get the duration
                import librosa
                y, sr = librosa.load(file.path)
                audio_length = librosa.get_duration(y=y, sr=sr)
            except:
                raise Exception("Could not get the duration of the audio file. Your file may be corrupted or in an unsupported format.")

        min_silence_length = float(min_silence_length)
        min_segment_length = float(min_segment_length)
        if min_segment_length < 0:
            min_segment_length = audio_length / 20
            min_segment_length = max(min_segment_length, 15.0)

        audio_path = file.path
        if backend == "whisperx":
            whisper = sieve.function.get("sieve/whisperx")
        else:
            whisper = sieve.function.get("sieve/whisper")
        translate = sieve.function.get("sieve/seamless_text2text")

        # create a temporary directory to store the audio files
        import os

        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

        def process_segment(segment):
            import time
            t = time.time()
            start_time, end_time = segment

            whisper_job_args = {
                "audio": sieve.File(path=file.path),
                "language": source_language,
                "word_level_timestamps": word_level_timestamps,
                "speed_boost": speed_boost,
                "start_time": start_time,
                "end_time": end_time,
                "initial_prompt": initial_prompt,
            }
            if backend != "whisperx":
                whisper_job_args["decode_boost"] = (backend == "whisper-timestamped")
            
            whisper_job = whisper.push(**whisper_job_args)
            print(f"Took {time.time() - t:.2f} seconds to push segment from {start_time:.2f} to {end_time:.2f}")
            return whisper_job

        if chunks == "":
            if self.model is None or (not use_vad and not use_pyannote_segmentation):
                print("Splitting audio into segments using ffmpeg silence detection...")
                segments = self.split_silences_by_ffmpeg_detect(
                    audio_path,
                    min_silence_length=min_silence_length,
                    min_segment_length=min_segment_length,
                )
            elif use_vad:
                print("Splitting audio into segments using Silero VAD...")
                segments = self.split_silences_by_silero_vad(
                    audio_path,
                    min_silence_length=min_silence_length,
                    min_segment_length=min_segment_length,
                    vad_threshold=vad_threshold,
                )
            elif use_pyannote_segmentation:
                print("Splitting audio into segments using Pyannote segmentation...")
                # we already have the diarization job, so we can use it to segment the audio
                diarization_job_output = diarization_job.result()
                segments = self.split_silences_by_pyannote(
                    audio_path,
                    diarization_job_output,
                    min_segment_length=min_segment_length,
                )
        else:
            try:
                # chunks is a string formatted with a start and end second on each line
                segments = [
                    tuple(map(float, line.split(",")))
                    for line in chunks.strip().split("\n")
                ]
            except:
                raise ValueError(
                    "Invalid chunks format. Please provide a string formatted with a start and end second on each line. Example: '0,10\n10,20\n20,30'"
                )
        if not segments:
            if not use_vad:
                whisper_job_args = {
                    "audio": sieve.File(path=file.path),
                    "language": source_language,
                    "word_level_timestamps": word_level_timestamps,
                    "speed_boost": speed_boost,
                    "initial_prompt": initial_prompt,
                }
                if backend != "whisperx":
                    whisper_job_args["decode_boost"] = (backend == "whisper-timestamped")
                
                segments.append(whisper.push(**whisper_job_args))
        job_outputs = []
        for job in executor.map(process_segment, segments):
            job_output = job.result()
            if job_output["text"] == "":
                continue
            job_segments = job_output["segments"]
            if len(job_segments) > 0:
                print(f"transcribed {100*job_segments[-1]['end'] / audio_length:.2f}% of {audio_length:.2f} seconds")
            if len(target_language) > 0 and job_output["language_code"] != target_language and job_output["text"]:
                if target_language not in WHISPER_TO_SEAMLESS_LANGUAGE_MAP:
                    raise Exception(
                        f"Target language not supported for translation: ",
                        target_language,
                    )
                if job_output["language_code"] not in WHISPER_TO_SEAMLESS_LANGUAGE_MAP:
                    raise Exception(
                        f"Detected language not supported for translation: ",
                        job_output["language_code"],
                    )

                # Output language is in Whisper's language coding, so we need to transform to seamless
                seamless_target_lang = WHISPER_TO_SEAMLESS_LANGUAGE_MAP[
                    target_language
                ]
                seamless_source_lang = WHISPER_TO_SEAMLESS_LANGUAGE_MAP[
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
        
        if len(job_outputs) == 0:
            # yield an empty segment if no segments were transcribed
            job_outputs.append({"text": "", "language_code": source_language, "segments": []})
            if not speaker_diarization:
                yield job_outputs[0]
        if speaker_diarization:
            diarization_job_output = diarization_job.result()

            def join_words(words):
                text = ""
                for w in words:
                    stripped = w["word"].lstrip()
                    if stripped.endswith(" "):
                        text += stripped
                    else:
                        text += stripped + " "
                text = text.strip()
                return text
            
            def word_timestamp_to_speaker(start_time, end_time):
                # find the segment that contains the greatest proportion of the word
                max_proportion = 0
                speaker_id = None
                for seg in diarization_job_output:
                    seg_start, seg_end = seg["start"], seg["end"]
                    # calculate overlap between the word and the segment
                    overlap = max(0, min(end_time, seg_end) - max(start_time, seg_start))
                    proportion = overlap / max((end_time - start_time), 1e-6)
                    if proportion > max_proportion:
                        max_proportion = proportion
                        speaker_id = seg["speaker_id"]
                if speaker_id is None or max_proportion == 0:
                    # if its inside any segment, return that
                    for seg in diarization_job_output:
                        if seg["start"] <= start_time <= seg["end"]:
                            return seg["speaker_id"]
                    # find the closest segment
                    seg = min(diarization_job_output, key=lambda x: abs(x["start"] - start_time))
                    return seg["speaker_id"]
                return speaker_id
            
            print("diarization finished")
            for job_output in job_outputs:
                transcript_segments = job_output["segments"]
                new_transcript_segments = []
                for seg in transcript_segments:
                    last_speaker = -1
                    words_list = []
                    if 'words' in seg:
                        for word in seg['words']:
                            if 'start' in word and 'end' in word:
                                speaker = word_timestamp_to_speaker(word['start'], word['end'])
                                word["speaker"] = speaker
                                if (speaker != last_speaker and len(words_list) > 0) or (len(words_list) > 0 and word['start'] - words_list[-1]['end'] > 0.8):
                                    new_transcript_segments.append({
                                        "text": join_words(words_list),
                                        "speaker": last_speaker,
                                        "start": words_list[0]["start"],
                                        "end": words_list[-1]["end"],
                                        "words": words_list,
                                    })
                                    words_list = []
                                words_list.append(word)
                                last_speaker = speaker
                        if len(words_list) > 1:
                            new_transcript_segments.append({
                                "text": join_words(words_list),
                                "speaker": last_speaker,
                                "start": words_list[0]["start"],
                                "end": words_list[-1]["end"],
                                "words": words_list,
                            })
                        elif len(words_list) == 1:
                            # join with the previous segment
                            if new_transcript_segments and words_list[0]['start'] - new_transcript_segments[-1]['end'] <= 0.8 and new_transcript_segments[-1]['speaker'] == words_list[0]['speaker']:
                                words_list[0]["speaker"] = new_transcript_segments[-1]["speaker"]
                                if new_transcript_segments[-1]["text"].endswith(" "):
                                    new_transcript_segments[-1]["text"] += words_list[0].get("word", "")
                                else:
                                    new_transcript_segments[-1]["text"] += " " + words_list[0].get("word", "")
                                new_transcript_segments[-1]["end"] = words_list[0]["end"]
                                new_transcript_segments[-1]["words"].append(words_list[0])
                            else:
                                new_transcript_segments.append({
                                    "text": words_list[0].get("word", ""),
                                    "speaker": words_list[0].get("speaker", -1),
                                    "start": words_list[0]["start"],
                                    "end": words_list[0]["end"],
                                    "words": words_list,
                                })
                job_output["segments"] = new_transcript_segments
                # add a "speaker" field to each segment
                for seg in job_output["segments"]:
                    seg["speaker"] = seg["words"][0]["speaker"]
                    # edge case when a segment is very short
                    if seg["start"] == seg["end"]:
                        seg["end"] = seg["start"] + 0.01
                
                # now combine segments where the adjacent segments have the same speaker and are within 0.8s of each other
                new_transcript_segments = []
                for i, seg in enumerate(job_output["segments"]):
                    if i == 0:
                        new_transcript_segments.append(seg)
                    else:
                        time_difference = seg["start"] - new_transcript_segments[-1]["end"]
                        if seg["speaker"] == new_transcript_segments[-1]["speaker"] and time_difference <= 0.8:
                            if new_transcript_segments[-1]["text"].endswith(" "):
                                new_transcript_segments[-1]["text"] += seg["text"]
                            else:
                                new_transcript_segments[-1]["text"] += " " + seg["text"]
                            new_transcript_segments[-1]["end"] = seg["end"]
                            new_transcript_segments[-1]["words"] += seg["words"]
                        else:
                            new_transcript_segments.append(seg)
                job_output["segments"] = new_transcript_segments
                yield job_output
        
        print("transcription finished")

if __name__ == "__main__":
    import subprocess
    file = sieve.File(path="/Users/Mokshith/Downloads/tmp9dwy6wqa.mp4")
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", f"{file.path}"], capture_output=True, text=True)
    print(result.stdout)

