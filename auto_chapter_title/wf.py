import sieve
from typing import Dict, List

@sieve.Model(
    name="whisper_full",
    python_packages=["git+https://github.com/openai/whisper.git"],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models",
        "wget -c 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt' -P /root/.cache/models"
    ]
)
class Whisper:
    def __setup__(self):
        import whisper
        self.model = whisper.load_model("/root/.cache/models/tiny.en.pt")

    def __predict__(self, audio: sieve.Audio) -> Dict:
        result = self.model.transcribe(audio.path)
        return {
            'text': result['text'],
            'segments': [
                {
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                } for segment in result['segments']
            ]
        }

@sieve.Model(
    name="text_tiling",
    python_packages=["nltk==3.6.2"],
    python_version="3.8",
)
class TextTiling:
    def __setup__(self):
        import nltk
        from nltk.tokenize import PunktSentenceTokenizer, TextTilingTokenizer
        nltk.download('stopwords')
        self.sentence_tokenizer = PunktSentenceTokenizer()
        # TODO: alter w based on length of transcript
        self.tokenizer = TextTilingTokenizer(w=50)

    def __predict__(self, transcript: Dict) -> Dict:
        import nltk
        from nltk.tokenize import PunktSentenceTokenizer, TextTilingTokenizer
        sentences = self.sentence_tokenizer.tokenize(transcript['text'])
        paragraphs = ['\n'.join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        sections = self.tokenizer.tokenize('\n\n'.join(paragraphs))

        for section in sections:
            start, end = float('inf'), 0
            for segment in transcript['segments']:
                if segment['text'].strip() in section:
                    start = min(start, segment['start'])
                    end = max(end, segment['end'])
            yield { 'text': section, 'start': start, 'end': end }

@sieve.function(
    name="generate_chapters",
    python_packages=["openai==0.26.5", "python-dotenv==0.21.1"]
)
def generate_chapters(segment: Dict) -> Dict:
    import os
    import openai
    from dotenv import load_dotenv
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    PROMPT = f"Here's a section of transcript from a Youtube video. Can you name this section?\nTranscript:{segment['text']}\nChapter:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=PROMPT,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    yield {
        "chapter": response.choices[0].text,
        "start": segment["start"],
        "end": segment["end"]
    }

@sieve.function(
    name="consolidate_chapters",
    python_packages=["openai==0.26.5", "python-dotenv==0.21.1"],
    iterator_input=True
)
def consolidate_chapters(chapters: Dict) -> List[Dict]:
    chapters = list(chapters)
    
    import os
    import openai
    import json
    from dotenv import load_dotenv
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    PROMPT = "Here are chapter titles for a Youtube video and their timestamps. Can you consolidate them to less than 5 chapters with their merged timestamps and remove any off-topic ones? Your output should be a list of JSON dictionaries, each with the chapter name, start and end time. Here's an example chapter in JSON: {\"chapter_name\": \"AI is taking over\", \"start_time\": 400.55, \"end_time\": 500.67}\n\nChapters:"

    for chapter in chapters:
        PROMPT += f"\n{chapter['start']}-{chapter['end']}: {chapter['chapter']}"
    
    PROMPT += "\n\nConsolidated Chapters:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=PROMPT,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    try: 
        return json.loads(response.choices[0].text)
    except json.decoder.JSONDecodeError:
        return [{"chapters": response.choices[0].text}]

@sieve.function(name="split_audio", python_packages=["moviepy==1.0.3"])
def split_audio(vid: sieve.Video) -> sieve.Audio:
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(vid.path)
    video.audio.write_audiofile("audio.mp3")
    return sieve.Audio(path="audio.mp3")

@sieve.workflow(name="auto_chapter_title")
def auto_chapter_title(vid: sieve.Video) -> List[Dict]:
    audio = split_audio(vid)
    text = Whisper()(audio)
    sections = TextTiling()(text)
    all_chapters = generate_chapters(sections)
    return consolidate_chapters(all_chapters)
