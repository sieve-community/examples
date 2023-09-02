import sieve
from typing import Dict, List

@sieve.function(
    name="transcript_analyzer",
    python_packages=[
        "gpt-json"
    ],
    python_version="3.10",
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key"),
        sieve.Env(name="NUM_TAGS", description="Number of tags to generate", default="5"),
        sieve.Env(name="MAX_SUMMARY_SENTENCES", description="Number of sentences to summarize", default="5"),
        sieve.Env(name="MAX_TITLE_WORDS", description="Max number of words in title", default="10"),
    ]
)
def analyze_transcript(transcript):
    yield {"transcript": transcript}
    import time
    overall_start_time = time.time()
    import os
    import json
    import asyncio

    from transcript_analysis import description_runner, chapter_runner

    max_num_sentences = int(os.getenv("MAX_SUMMARY_SENTENCES", 5))
    max_num_words = int(os.getenv("MAX_TITLE_WORDS", 10))
    num_tags = int(os.getenv("NUM_TAGS", 5))
    summary, title, tags = asyncio.run(description_runner(transcript, max_num_sentences=max_num_sentences, max_num_words= max_num_words, num_tags=num_tags))
    yield {"summary": summary}
    yield {"title": title}
    yield {"tags": tags}


    chapters = asyncio.run(chapter_runner(transcript))
    out_list = []
    for chapter in chapters:
        out_list.append({"title": chapter.title, "start_time": chapter.start_time})
    yield {"chapters": out_list}

@sieve.function(name="extract_audio", system_packages=["ffmpeg"])
def split_audio(vid: sieve.Video) -> sieve.Audio:
    import time
    overall_start_time = time.time()
    import subprocess
    video_path = vid.path

    audio_path = 'temp.wav'

    # overwrite existing file and turn into wav
    subprocess.run(["ffmpeg", "-i", video_path, audio_path, "-y"])
    print(f"Overall time: {time.time() - overall_start_time}")
    return sieve.Audio(path=audio_path)

wf_metadata = sieve.Metadata(
    title="Generate Video Title, Chapters, and Summary",
    description="Given a video, generate a title, chapters, and summary.",
    code_url="https://github.com/sieve-community/examples/tree/main/video_transcript_analysis/main.py",
    tags=["Video"],
    image=sieve.Image(
        url="https://www.tubebuddy.com/wp-content/uploads/2022/06/video-chapter-snippet-1024x674.png"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.workflow(name="video_transcript_analysis", metadata=wf_metadata)
def auto_chapter_title(vid: sieve.Video) -> List[Dict]:
    """
    :param vid: A video to transcribe and split into chapters
    :return: A list of chapter titles with their start and end times
    """
    audio = split_audio(vid)
    text = sieve.reference("sieve/whisperx")(audio)
    analysis = analyze_transcript(text)
    return analysis


if __name__ == "__main__":
    sieve.push(
        workflow="video_transcript_analysis",
        inputs={
            "vid": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/The%20Truth%20About%20AI%20Getting%20Creative.mp4"
            }
        },
    )
