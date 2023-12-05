import sieve

cloning_metadata = sieve.Metadata(
    description="Clone a voice using ElevenLabs",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_speech/elevenlabs",
    image=sieve.Image(
        url="https://yt3.googleusercontent.com/Z1wLpOYEwoXNLqUI9zhmnir841WhkJGV-HdRbWGdlnK6FaHNlRHCMulzjF3dmm5y8Um-laxKORQ=s900-c-k-c0x00ffffff-no-rj"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning"],
    readme=open("CLONING_README.md", "r").read(),
)

@sieve.function(
    name="elevenlabs_voice_cloning",
    system_packages=["ffmpeg"],
    environment_variables=[
        sieve.Env(name="ELEVEN_LABS_API_KEY", description="API key for ElevenLabs")
    ],
    metadata=cloning_metadata
)
def clone_audio(
    reference_audio: sieve.Audio,
    delete_voice_id: str = ""
):
    '''
    :param reference_audio: reference audio to use when cloning voice
    :param delete_voice_id: if set, function will delete the specified voice ID instead of cloning a voice and returning the ID
    :return: An JSON payload with the voice ID
    '''
    import os
    import subprocess
    import tempfile
    import requests
    
    API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")
    if API_KEY == "":
        raise Exception("ELEVEN_LABS_API_KEY is not set as an env var")

    if delete_voice_id and len(delete_voice_id) > 0:
        try:
            url = f"https://api.elevenlabs.io/v1/voices/{delete_voice_id}"

            headers = {"Accept": "application/json", "xi-api-key": API_KEY}

            response = requests.delete(url, headers=headers)
            return response.json()
        except:
            print(f"Unable to delete voice id: {delete_voice_id}")
            print(response.text)
            return response.json()
        
    SOURCE_AUDIO_PATH = reference_audio.path

    # Check if audio is already in wav format
    if not SOURCE_AUDIO_PATH.endswith('.wav'):
        # Convert audio to wav
        wav_path = "temp.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                SOURCE_AUDIO_PATH,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                wav_path,
            ]
        )
        SOURCE_AUDIO_PATH = wav_path

    temp_dir = tempfile.mkdtemp()

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            SOURCE_AUDIO_PATH,
            "-f",
            "segment",
            "-segment_time",
            "10",
            "-c",
            "copy",
            f"{temp_dir}/%03d.wav",
        ]
    )

    wav_paths = [
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".wav")
    ]
    wav_paths.sort()

    url = "https://api.elevenlabs.io/v1/voices/add"

    headers = {"Accept": "application/json", "xi-api-key": API_KEY}

    import uuid

    data = {
        "name": str(uuid.uuid4()),
        "description": "Voice description",
    }

    files = [
        ("files", (fname.split("/")[-1], open(fname, "rb"), "audio/wav"))
        for fname in wav_paths
    ]
    # cap to 20 files
    files = files[:15]

    response = requests.post(url, headers=headers, data=data, files=files)
    if response.status_code != 200:
        raise ValueError("Could not generate voice", response.text)
    print(response.text)

    # delete temp folder
    import shutil

    shutil.rmtree(temp_dir)

    try:
        return response.json()
    
    except Exception as e:
        print(response.text)
        raise ValueError("Could not generate voice", response.text)

synthesis_metadata = sieve.Metadata(
    description="Text to speech using ElevenLabs",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_speech/elevenlabs",
    image=sieve.Image(
        url="https://yt3.googleusercontent.com/Z1wLpOYEwoXNLqUI9zhmnir841WhkJGV-HdRbWGdlnK6FaHNlRHCMulzjF3dmm5y8Um-laxKORQ=s900-c-k-c0x00ffffff-no-rj"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning"],
    readme=open("TTS_README.md", "r").read(),
)

@sieve.function(
    name="elevenlabs_speech_synthesis",
    system_packages=["ffmpeg"],
    environment_variables=[
        sieve.Env(name="ELEVEN_LABS_API_KEY", description="API key for ElevenLabs")
    ],
    metadata=synthesis_metadata
)
def generate_audio(
    text: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    stability: float = 0.5,
    similarity_boost: float = 0.63,
    style: float = 0.0,
    use_speaker_boost: bool = True,
    model_id: str = "eleven_multilingual_v2",
) -> sieve.Audio:
    """
    :param text: text to speak
    :param voice_id: the ID of the ElevenLabs voice to use. If none are set, the ID of Rachel will be used.
    :param stability: Value between 0 and 1. Increasing variability can make speech more expressive with output varying between re-generations. It can also lead to instabilities.
    :param similarity_boost: Value between 0 and 1. Low values are recommended if background artifacts are present in generated speech.
    :param style: Value between 0 and 1. High values are recommended if the style of the speech should be exaggerated compared to the uploaded audio. Higher values can lead to more instability in the generated speech. Setting this to 0.0 will greatly increase generation speed and is the default setting.
    :param use_speaker_boost: Boost the similarity of the synthesized speech and the voice at the cost of some generation speed.
    :param model_id: pick between eleven_multilingual_v2, eleven_multilingual_v1, and eleven_monolingual_v1
    :return: A generated speech file
    """

    import os
    
    API_KEY = os.environ.get("ELEVEN_LABS_API_KEY", "")
    if API_KEY == "":
        raise Exception("ELEVEN_LABS_API_KEY is not set as an env var")

    import requests

    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY,
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        },
    }

    # check if output file exists and delete it
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")

    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise ValueError("Could not generate voice", response.text)
    with open("output.mp3", "wb") as f:
        f.write(response.content)

    return sieve.Audio(path="output.mp3")