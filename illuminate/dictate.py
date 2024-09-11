import os
import shutil
import random

import sieve
from langchain.chat_models import ChatOpenAI
from pydub import AudioSegment

VOICE_DIR = "voices"

EMOTIONS = [
    "normal",
    "anger",
    "curiosity",
    "positivity",
    "surprise",
    "sadness",
]

PACES = [
    "normal",
    "fast",
    "slow"
]


llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")


def parse_emotion_pace(text):
    prompt = lambda message: f"""
Which of the emotions:
- {"\n- ".join(EMOTIONS)}

And paces:
- {"\n- ".join(PACES)}

Is most apprporiate to read this:
{message} 
"""
    response = llm.invoke(prompt(text)).content
    words = response.lower().split(" ")

    emotion = "normal"
    pace = "normal"

    for word in words:
        if word in EMOTIONS:
            emotion = word
        if word in PACES:
            pace = word

    return {"emotion": emotion, "pace": pace}


def intonate(messages):

    for message in messages:
        text = message['message']
        message = {**message, **parse_emotion_pace(text)}

        yield message


def dictate_messages(
        messages: dict, 
        alice: str="obama", 
        bob: str="neil_degrasse"
):

    tts = sieve.function.get("sieve/tts")
    backend = "cartesia-voice-cloning"

    ref_voices = {
        "user": sieve.File(path=os.path.join(VOICE_DIR, alice + ".wav")),
        "assistant": sieve.File(path=os.path.join(VOICE_DIR, bob + ".wav")),
    }

    voice_jobs = []
    for message in intonate(messages):
        if message["role"] == "system":
            continue

        args = {
            "voice": backend,
            "text": message["message"],
            "reference_audio": ref_voices[message["role"]],
            "emotion": message["emotion"],
            "pace": message["pace"],
        }

        voice_jobs.append(
            tts.push(**args)
        )

    for job in voice_jobs:
        yield job.result()


def concatenate_audio_files(directory):
    print(f"Concatenating audio files in {directory}")
    output_file = "concatenated.wav"
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    audio_files.sort()

    combined = AudioSegment.empty()

    for file in audio_files:
        audio = AudioSegment.from_wav(os.path.join(directory, file))

        # add pause of around half a second; noise of 50ms makes it sound more natural
        noise = random.gauss(0,50)
        pause = AudioSegment.silent(duration=500 + noise)

        combined += audio + pause

    combined.export(output_file, format="wav")

    return sieve.File(path=output_file)


def main(messages):
    convo_dir = "convo"

    shutil.rmtree(convo_dir, ignore_errors=True)
    os.makedirs(convo_dir, exist_ok=True)
    for i, audio in enumerate(dictate_messages(messages)):
        shutil.move(audio.path, os.path.join("convo", f"audio_{i}.wav"))
        print(f"Audio {i} saved to convo/audio_{i}.wav")

    return concatenate_audio_files(convo_dir)


if __name__ == "__main__":

    messages = [
        {"role": "user", "message": "Hello, how are you?"},
        {"role": "assistant", "message": "I'm doing well, thank you for asking."},
        {"role": "user", "message": "What is the capital of France?"},
        {"role": "assistant", "message": "The capital of France is Paris."},
    ]


    main(messages)



