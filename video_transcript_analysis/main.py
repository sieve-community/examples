# TODO: change this thing

import sieve
from typing import Dict, List
import tempfile

metadata = sieve.Metadata(
    title="Analyze Transcripts",
    description="Given a video or audio, generate a title, chapters, summary, tags, and highlights.",
    code_url="https://github.com/sieve-community/examples/tree/main/video_transcript_analysis/main.py",
    tags=["Video", "Featured", "Transcription"],
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/video_transcript_analyzer.jpg"
    ),
    readme=open("README.md", "r").read(),
)


@sieve.function(
    name="video_transcript_analyzer",
    python_packages=["gpt-json>=0.4.2", "numpy"],
    system_packages=["ffmpeg"],
    python_version="3.11",
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key")
    ],
    metadata=metadata,
)
def analyze_transcript(
    file: sieve.File,
    generate_chapters: bool = True,
    generate_highlights: bool = False,
    max_summary_length: int = 5,
    max_title_length: int = 10,
    num_tags: int = 5,
    denoise_audio: bool = True,
    highlight_search_phrases : str = "Most likely to go viral, funniest",
    max_highlight_duration: int = 30,
):
    '''
    :param file: Video or audio file
    :param max_summary_length: Maximum number of sentences in summary. Defaults to 5.
    :param max_title_length: Maximum number of words in title. Defaults to 10.
    :param num_tags: Number of tags to generate. Defaults to 5.
    :param generate_chapters: Whether to generate chapters or not. Defaults to True.
    :param denoise_audio: Whether to denoise audio before analysis. Results in better transcription but slower processing. Defaults to True.
    :param generate_highlights: Whether to generate highlights or not. Defaults to False.
    :param highlight_search_phrases: Topic(s) of highlights to generate, can be multiple comma-separated phrases. Can be anything from "Most likely to go viral" to "Technology". Defaults to "Most likely to go viral, funniest".
    :param max_highlight_duration: Maximum duration of each highlight in seconds.
    '''
    print("converting to audio")
    # video to audio
    import subprocess
    import os
    import uuid

    # Extract the length of the video using ffprobe
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file.path], capture_output=True, text=True)
    video_length = float(result.stdout)

    id = str(uuid.uuid4())
    audio_path = "temp" + id + ".wav"
    try:
        subprocess.run(["ffmpeg", "-i", file.path, audio_path, "-y"])
    except Exception as e:
        raise Exception(
            "Failed to extract audio from video. Make sure video has audio."
        )

    if not os.path.isfile(audio_path):
        raise Exception(
            "Failed to extract audio from video. Make sure video has audio."
        )

    print("conversion finished")
    print("running speech to text")
    # audio to text
    whisper = sieve.function.get("sieve/speech_transcriber")
    transcript = []
    transcript_segments = []
    for transcript_chunk in whisper.run(sieve.File(path=audio_path), denoise_audio=denoise_audio, min_segment_length = max_highlight_duration*2, use_vad = True, initial_prompt = "I made sure to add full capitalization and punctuation."):
        transcript.append(transcript_chunk)
        segments = transcript_chunk["segments"]
        transcript_segments.append(segments)
        if len(segments) > 0:
            print(f"transcribed {100*segments[-1]['end'] / video_length:.2f}% of {video_length:.2f} seconds")

    language_code = transcript[0]["language_code"]
    # flatten transcript into single list. right now it is a list of list of segments
    print("speech to text finished")

    transcript = [segment["segments"] for segment in transcript]
    transcript = [item for sublist in transcript for item in sublist]
    average_confidence_per_segment = []
    for segment in transcript:
        average_confidence_per_segment.append(
            sum([word["confidence"] for word in segment["words"]]) / len(segment["words"])
        )
    
    # count segments with low confidence
    num_low_confidence_segments = sum([1 for confidence in average_confidence_per_segment if confidence < 0.5])

    text = "".join([word["word"] for segment in transcript for word in segment["words"]])
    yield {"text": text, "language_code": language_code, "media_length_seconds": video_length}
    yield {"transcript": transcript}

    if len(text.split(" ")) < 15 or num_low_confidence_segments > 0.25 * len(transcript):
        yield {"summary": "No summary available. Video is too short, has no audio, or has too many low confidence segments."}
        yield {"title": "No title available. Video is too short, has no audio, or has too many low confidence segments."}
        yield {"tags": []}
        if generate_chapters:
            yield {"chapters": []}
        return

    import os
    import asyncio
    from datetime import timedelta
    from transcript_analysis import description_runner, chapter_runner, highlight_runner, compute_scores, seconds_to_timestamp

    transcript_segments = [item for sublist in transcript_segments for item in sublist]
    
    extended_dict = {
        index: {
            'text': segment['text'],
            'duration': segment['end'] - segment['start'],
            'start_time': segment['start'],
            'end_time': segment['end'],
            'score': 0 
        } for index, segment in enumerate(transcript_segments)
    }
    if generate_highlights:
        print("running highlight runner")
        scores = asyncio.run(highlight_runner([segment['text'] for segment in extended_dict.values()], highlight_search_phrases))

    max_num_sentences = max_summary_length
    max_num_words = max_title_length
    num_tags = num_tags
    print("running description runner")
    summary, title, tags = asyncio.run(
        description_runner(
            transcript,
            max_num_sentences=max_num_sentences,
            max_num_words=max_num_words,
            num_tags=num_tags,
        )
    )
    print("finished description runner")
    yield {"summary": summary}
    yield {"title": title}
    yield {"tags": tags}
    
    if generate_highlights:
        optimal_windows = compute_scores(extended_dict, scores, max_highlight_duration, summary)
        yield {"highlights": optimal_windows}

    print("running chapter runner")

    if generate_chapters:
        chapters = asyncio.run(chapter_runner(transcript))
        print("finished chapter runner")
        out_list = []
        for chapter in chapters:
            out_list.append({"title": chapter["title"], "start_time": chapter["start_time"], "timecode": chapter["timecode"]})
        yield {"chapters": out_list}

    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    for out in analyze_transcript.run(sieve.File(url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/702b88c8-d5f9-42bb-9fa5-5bce2e4b96ee/c127322c-a884-4fe9-97b9-de38e58f78dd-input-file.mp4")):
        print(out)
