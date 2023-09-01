import sieve
from typing import Dict, List

@sieve.Model(
    name="transcript_analyzer",
    python_packages=[
        "nltk==3.6.2",
        "openai==0.27.9",
        "python-dotenv==0.21.1"
    ],
    python_version="3.8",
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key"),
        sieve.Env(name="NUM_CHAPTERS", description="Number of chapters to generate", default="5"),
        sieve.Env(name="SUMMARY_NUM_SENTENCES", description="Number of sentences to summarize", default="5"),
    ]
)
class TranscriptAnalyzer:
    def __setup__(self):
        import nltk
        from nltk.tokenize import PunktSentenceTokenizer, TextTilingTokenizer

        nltk.download("stopwords")
        self.sentence_tokenizer = PunktSentenceTokenizer()
        # TODO: alter w based on length of transcript
        self.tokenizer = TextTilingTokenizer(w=50)

    def __predict__(self, transcript) -> Dict:
        import time
        overall_start_time = time.time()
        import nltk
        from nltk.tokenize import PunktSentenceTokenizer, TextTilingTokenizer
        import os
        import openai
        import json

        openai.api_key = os.getenv("OPENAI_API_KEY")

        '''
        Given a transcript, summarize it.
        '''
        text = [segment["text"] for segment in transcript]
        text = " ".join(text)

        num_sentences = int(os.getenv("SUMMARY_NUM_SENTENCES", "5"))

        PROMPT = f"Here's a transcript from a video. Can you summarize this section in {num_sentences} sentences or less? Please summarize it in a 3rd person style, where this summary is read as a description of a video that might come after the phrase, 'In this video...'.\nTranscript:{text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that is great at summarizing and chapterizing video transcripts."},
                {"role": "user", "content": PROMPT},
            ]
        )

        yield {"summary": response['choices'][0]['message']['content'].strip().replace("\n", "")}

        PROMPT = f"Here's a transcript from a video. Can you come up with a title for this transcript? Make it about the content that's being spoken about rather than who is saying the content. Respond with nothing but the title text. \nTranscript:{text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that is great at summarizing and chapterizing video transcripts."},
                {"role": "user", "content": PROMPT},
            ]
        )

        yield {"title": response['choices'][0]['message']['content'].strip().replace('\"', '')}

        '''
        Given a transcript, generate chapters.
        '''

        sentences = self.sentence_tokenizer.tokenize(text)
        paragraphs = [
            "\n".join(sentences[i : i + 2]) for i in range(0, len(sentences), 2)
        ]
        sections = self.tokenizer.tokenize("\n\n".join(paragraphs))
        chapters = []
        for section in sections:
            start, end = float("inf"), 0
            for segment in transcript:
                if segment["text"].strip() in section:
                    start = min(start, segment["start"])
                    end = max(end, segment["end"])
            seg = {"text": section, "start": start, "end": end}


            PROMPT = f"Here's a section of transcript from a Youtube video. Can you name this section?\nTranscript:{seg['text']}\nChapter:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that is great at summarizing and chapterizing video transcripts."},
                    {"role": "user", "content": PROMPT},
                ]
            )
            chapters.append({
                "chapter": response['choices'][0]['message']['content'],
                "start": seg["start"],
                "end": seg["end"],
            })

        num_chapters = int(os.getenv("NUM_CHAPTERS", "5"))

        PROMPT = f"Here are chapter titles for a Youtube video and their timestamps. Can you consolidate them to less than {num_chapters} "+ 'chapters with their merged timestamps and remove any off-topic ones? Your output should be a list of JSON dictionaries, each with the chapter name, start and end time. Here\'s an example chapter in JSON: {"name": "AI Cannot Replace Online Creators", "start_time": 400.55, "end_time": 500.67}\n\nChapters:'

        for chapter in chapters:
            PROMPT += f"\n{chapter['start']}-{chapter['end']}: {chapter['chapter']}"

        PROMPT += "\n\nConsolidated Chapters:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that is great at summarizing and chapterizing video transcripts."},
                {"role": "user", "content": PROMPT},
            ]
        )

        try:
            yield {
                "chapters": json.loads(response['choices'][0]['message']['content']),
            }
        except json.decoder.JSONDecodeError:
            yield {"chapters": response['choices'][0]['message']['content']}

        print(f"Overall time: {time.time() - overall_start_time}")

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
    analysis = TranscriptAnalyzer()(text)
    return analysis, text


if __name__ == "__main__":
    sieve.push(
        workflow="auto_chapter_title",
        inputs={
            "vid": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/The%20Truth%20About%20AI%20Getting%20Creative.mp4"
            }
        },
    )
