import sieve
import os
import shutil
import subprocess
import tempfile
import json


VOICE_DIR = "voices"

ALL_VOICES = [
    {
        "name": "neil_degrasse",
        "link": "https://www.youtube.com/watch?v=JtahB1-MNvk",
        "time": 2*60 + 14
    },
    {
        "name": "obama",
        "link": "https://www.youtube.com/watch?v=X15o2sG8HF4",
        "time": 1*60 + 5
    },
    {
        "name": "oprah",
        "link": "https://www.youtube.com/watch?v=Nk3s5SvIQ7o",
        "time": 1*60 + 40
    },
]


def save_voice(yt_link: str, name: str):
    os.makedirs(VOICE_DIR, exist_ok=True)

    youtube_dl = sieve.function.get("sieve/youtube_to_mp4")
    video = youtube_dl.run(yt_link, resolution="lowest-available")

    shutil.move(video.path, os.path.join(VOICE_DIR, name + ".mp4"))


def trim_video_to_wav(video: sieve.File, start_time: int):
    # tmp wav file w random name
    output_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".wav")

    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-t', '60',   # 60 seconds
        '-i', video.path,
        '-nostdin',
        '-loglevel','error',
        '-vn',   # Disable video output
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '44100',   # Audio sampling rate
        '-ac', '2',   # Stereo
        output_path
    ]

    subprocess.run(command, check=True)

    return sieve.File(path=output_path)


def get_voices():

    os.makedirs(VOICE_DIR, exist_ok=True)

    youtube_dl = sieve.function.get("sieve/youtube_to_mp4")

    jobs = []
    for voice in ALL_VOICES:
        if os.path.exists(os.path.join(VOICE_DIR, voice["name"] + ".wav")):
            print(f"Voice {voice['name']} already exists")
            continue

        jobs.append(youtube_dl.push(
            voice["link"],
            resolution="lowest-available"
        ))
        print(f"Downloading {voice['name']} voice")


    for job, voice in zip(jobs, ALL_VOICES):

        video = job.result()

        audio = trim_video_to_wav(video, voice["time"])

        shutil.move(audio.path, os.path.join(VOICE_DIR, voice["name"] + ".wav"))
        print(f"Saved {voice['name']} voice to {VOICE_DIR}")



if __name__ == "__main__":
    get_voices()
