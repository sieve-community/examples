#TODO: change this thing

import sieve
from typing import Dict, List

metadata = sieve.Metadata(
    title="Video Transcript Analysis",
    description="Given a video or audio, generate a title, chapters, summary and tags",
    code_url="https://github.com/sieve-community/examples/tree/main/video_transcript_analysis/main.py",
    tags=["Video", "Featured", "Transcription"],
    image=sieve.Image(
        url="https://www.tubebuddy.com/wp-content/uploads/2022/06/video-chapter-snippet-1024x674.png"
    ),
    readme=open("README.md", "r").read(),
)


@sieve.function(
    name="video_transcript_analyzer",
    python_packages=[
        "gpt-json"
    ],
    system_packages=[
        "ffmpeg"
    ],
    python_version="3.10",
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key"),
        sieve.Env(name="NUM_TAGS", description="Number of tags to generate", default="5"),
        sieve.Env(name="MAX_SUMMARY_SENTENCES", description="Number of sentences to summarize", default="5"),
        sieve.Env(name="MAX_TITLE_WORDS", description="Max number of words in title", default="10"),
    ],
    metadata=metadata
)
def analyze_transcript(file: sieve.Video) -> Dict:
    print("running ffmpeg to convert video to audio")
    # video to audio
    import subprocess
    import os
    try:
        audio_path = 'temp.wav'
        subprocess.run(["ffmpeg", "-i", file.path, audio_path, "-y"])
    except Exception as e:
        raise Exception("Failed to extract audio from video. Make sure video has audio.")
    
    if not os.path.isfile(audio_path):
        raise Exception("Failed to extract audio from video. Make sure video has audio.")
    
    print("ffmpeg finished")
    print("running speech to text")
    # audio to text
    whisper = sieve.function.get("sieve/speech_transcriber")
    transcript = list(whisper.run(sieve.Audio(path=audio_path)))

    language_code = transcript[0]["language_code"]
    # flatten transcript into single list. right now it is a list of list of segments
    print("speech to text finished")
    text = " ".join([segment["text"] for segment in transcript]).strip()
    yield {"text": text, "language_code": language_code}

    transcript = [segment["segments"] for segment in transcript]
    transcript = [item for sublist in transcript for item in sublist]
    yield {"transcript": transcript}

    if len(text.split(" ")) < 15:
        yield {"summary": "No summary available. Video is too short or has no audio."}
        yield {"title": "No title available. Video is too short or has no audio."}
        yield {"tags": []}
        yield {"chapters": []}
        return

    import time
    overall_start_time = time.time()
    import os
    import json
    import asyncio

    from transcript_analysis import description_runner, chapter_runner

    max_num_sentences = int(os.getenv("MAX_SUMMARY_SENTENCES", 5))
    max_num_words = int(os.getenv("MAX_TITLE_WORDS", 10))
    num_tags = int(os.getenv("NUM_TAGS", 5))
    print("running description runner")
    summary, title, tags = asyncio.run(description_runner(transcript, max_num_sentences=max_num_sentences, max_num_words= max_num_words, num_tags=num_tags))
    print("finished description runner")
    yield {"summary": summary}
    yield {"title": title}
    yield {"tags": tags}

    print("running chapter runner")
    chapters = asyncio.run(chapter_runner(transcript))
    print("finished chapter runner")
    out_list = []
    for chapter in chapters:
        out_list.append({"title": chapter.title, "start_time": chapter.start_time})
    yield {"chapters": out_list}
