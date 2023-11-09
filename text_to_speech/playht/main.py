import sieve

cloning_metadata = sieve.Metadata(
    description="Clone a voice using Play.ht",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_speech/elevenlabs",
    image=sieve.Image(
        url="https://ps.w.org/play-ht/assets/icon-256x256.png?rev=1810827"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning"],
    readme=open("CLONING_README.md", "r").read(),
)

@sieve.function(
    name="playht_voice_cloning",
    system_packages=["ffmpeg"],
    environment_variables=[
        sieve.Env(name="PLAYHT_API_KEY", description="API key for Play.ht"),
        sieve.Env(name="PLAYHT_API_USER_ID", description="API user ID for Play.ht")
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
    
    API_KEY = os.environ.get("PLAYHT_API_KEY", "")
    USER_ID = os.environ.get("PLAYHT_API_USER_ID", "")
    if API_KEY == "":
        raise Exception("ELEVEN_LABS_API_KEY is not set as an env var")

    if delete_voice_id and len(delete_voice_id) > 0:
        try:
            url = f"https://api.play.ht/api/v2/cloned-voices"

            headers = {"Accept": "application/json", "AUTHORIZATION": API_KEY, "X-USER-ID": USER_ID, "content-type": "application/json"}
            payload = {"voice_id": delete_voice_id}

            response = requests.delete(url, headers=headers, json=payload)
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


    url = "https://api.play.ht/api/v2/cloned-voices/instant"

    headers = {"Accept": "application/json", "AUTHORIZATION": API_KEY, "X-USER-ID": USER_ID}

    import uuid

    data = {
        "voice_name": str(uuid.uuid4()),
    }

    files = { "sample_file": (SOURCE_AUDIO_PATH.split("/")[-1], open(SOURCE_AUDIO_PATH, "rb"), "audio/wav") }

    response = requests.post(url, headers=headers, data=data, files=files)
    print(response.text)

    try:
        return response.json()
    
    except Exception as e:
        print(response.text)
        raise ValueError("Could not generate voice", response.text)

synthesis_metadata = sieve.Metadata(
    description="Text to speech using Play.ht",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_speech/elevenlabs",
    image=sieve.Image(
        url="https://ps.w.org/play-ht/assets/icon-256x256.png?rev=1810827"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning"],
    readme=open("TTS_README.md", "r").read(),
)

@sieve.function(
    name="playht_speech_synthesis",
    system_packages=["ffmpeg"],
    environment_variables=[
        sieve.Env(name="PLAYHT_API_KEY", description="API key for Play.ht"),
        sieve.Env(name="PLAYHT_API_USER_ID", description="API user ID for Play.ht")
    ],
    metadata=synthesis_metadata
)
def generate_audio(
    text: str,
    voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
    quality: str = "high",
    speed: float = 1.0,
    seed: int = 0,
    temperature: float = 1.0,
    voice_engine: str = "PlayHT2.0",
    emotion: str = "",
    voice_guidance: float = 3,
    style_guidance: float = 20,
    text_guidance: float = 1
) -> sieve.Audio:
    """
    :param text: text to speak
    :param voice: The unique ID for a PlayHT or Cloned Voice.
    :param quality: The quality of the generated audio. Choose between "draft", "low", "medium", "high", and "premium". The default is "high".
    :param speed: Control how fast the generated audio should be. A number greater than 0 and less than or equal to 5.0.
    :param seed: An integer number greater than or equal to 0. If equal to null or not provided, a random seed will be used. Useful to control the reproducibility of the generated audio. Assuming all other properties didn't change, a fixed seed should always generate the exact same audio file.
    :param temperature: A floating point number between 0, inclusive, and 2, inclusive. If equal to null or not provided, the model's default temperature will be used. The temperature parameter controls variance. Lower temperatures result in more predictable results, higher temperatures allow each run to vary more, so the voice may sound less like the baseline voice.
    :param voice_engine: The voice engine used to synthesize the voice. Choose between "PlayHT2.0-turbo", "PlayHT2.0", and "PlayHT1.0". The default is "PlayHT2.0-turbo".
    :param emotion: An emotion to be applied to the speech. Only supported when voice_engine is set to PlayHT2.0 or PlayHT2.0-turbo, and voice uses that engine.
    :param voice_guidance: A number between 1 and 6. Use lower numbers to reduce how unique your chosen voice will be compared to other voices. Higher numbers will maximize its individuality. Only supported when voice_engine is set to PlayHT2.0 or PlayHT2.0-turbo, and voice uses that engine.
    :param style_guidance: A number between 1 and 30. Use lower numbers to to reduce how strong your chosen emotion will be. Higher numbers will create a very emotional performance. Only supported when voice_engine is set to PlayHT2.0 or PlayHT2.0-turbo, and voice uses that engine.
    :param text_guidance: A number between 1 and 2. This number influences how closely the generated speech adheres to the input text. Use lower values to create more fluid speech, but with a higher chance of deviating from the input text. Higher numbers will make the generated speech more accurate to the input text, ensuring that the words spoken align closely with the provided text. Only supported when voice_engine is set to PlayHT2.0, and voice uses that engine.
    :return: An audio file
    """

    import os
    
    API_KEY = os.environ.get("PLAYHT_API_KEY", "")
    USER_ID = os.environ.get("PLAYHT_API_USER_ID", "")
    if API_KEY == "":
        raise Exception("PLAYHT_API_KEY is not set as an env var")

    import requests

    CHUNK_SIZE = 1024
    url = "https://api.play.ht/api/v2/tts/stream"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "AUTHORIZATION": API_KEY,
        "X-USER-ID": USER_ID
    }

    data = {
        "text": text,
        "output_format": "mp3",
        "voice": voice,
        "quality": quality,
        "speed": speed,
        "seed": seed,
        "temperature": temperature,
        "voice_engine": voice_engine,
        "voice_guidance": voice_guidance,
        "style_guidance": style_guidance,
        "text_guidance": text_guidance
    }

    if len(emotion) > 0:
        data["emotion"] = emotion

    response = requests.post(url, json=data, headers=headers)
    with open("output.mp3", "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    return sieve.Audio(path="output.mp3")