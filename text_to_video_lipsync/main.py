import sieve

metadata = sieve.Metadata(
    description="Generate a video of a person speaking a given text using lipsyncing.",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_video_lipsync",
    image=sieve.Image(
        url="https://storage.googleapis.com/mango-public-models/dalle-lipsync-lego.png"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning", "Video", "Lipsyncing", "Featured"],
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="text_to_video_lipsync",
    system_packages=[
        "ffmpeg",
        "rubberband-cli"
    ],
    python_packages=[
        "librosa",
        "soundfile",
        "moviepy",
        "pydub",
        "pyrubberband",
    ],
    metadata=metadata,
    environment_variables=[
        sieve.Env(name="ELEVEN_LABS_API_KEY", description="API key for ElevenLabs", default=""),
        sieve.Env(name="PLAYHT_API_KEY", description="API key for ElevenLabs", default=""),
        sieve.Env(name="PLAYHT_API_USER_ID", description="API user ID for ElevenLabs", default="")
    ],
)
def do(
    source_video: sieve.Video,
    text: str,
    tts_model: str = "xtts",
    speech_stability: float = 0.5,
    speech_similarity_boost: float = 0.63,
    voice_id: str = "",
    cleanup_voice_id: bool = False,
    refine_source_audio: bool = True,
    refine_target_audio: bool = True,
    low_resolution: bool = False,
    low_fps: bool = False,
):
    '''
    :param source_video: video to lip-sync
    :param text: text to speak
    :param tts_model: TTS model to use. Supported models: "xtts", "elevenlabs", or "playht". "elevenlabs" or "playht" is recommended for better quality but requires an API key.
    :param speech_stability: Value between 0 and 1. Increasing variability can make speech more expressive with output varying between re-generations. It can also lead to instabilities.
    :param speech_similarity_boost: Value between 0 and 1. Low values are recommended if background artifacts are present in generated speech.
    :param voice_id: The ID of the ElevenLabs or Play.ht voice to use. If none are set, the voice will be cloned from the source audio and used. Only applicable if tts_model is "elevenlabs" or "playht".
    :param cleanup_voice_id: Whether to delete the voice after use. Only applicable if tts_model is "elevenlabs" or "playht".
    :param refine_source_audio: Whether to refine the source audio using sieve/audio_enhancement.
    :param refine_target_audio: Whether to refine the generated target audio using sieve/audio_enhancement.
    :param low_resolution: Whether to reduce the resolution of the output video to half of the original on each axis; significantly speeds up inference.
    :param low_fps: Whether to reduce the fps of the output video to half of the original; significantly speeds up inference.
    :return: A generated video file
    '''
    import os
    import time
    from utils import extract_audio_from_video, trim_silence_from_video, trim_silence_from_audio_loaded, convert_to_format

    source_video_path = source_video.path
    source_audio_path = "source_audio.wav"
    if os.path.exists(source_audio_path):
        os.remove(source_audio_path)
    
    # Extract audio from source_video
    extract_audio_from_video(source_video_path, source_audio_path)
    source_audio = sieve.Audio(path=source_audio_path)

    # Refine source_audio
    if refine_source_audio:
        start_time = time.time()
        if tts_model == "xtts":
            source_audio = sieve.function.get("sieve/audio_enhancement").run(source_audio, filter_type="all")
        elif (tts_model == "elevenlabs" and len(voice_id) == 0):
            source_audio = sieve.function.get("sieve/audio_enhancement").run(source_audio, filter_type="noise")
        print(f"Time taken to refine source audio: {time.time() - start_time} seconds")    

    # split text into sentences by punctuation
    import re
    segments = []
    for sentence in re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text):
        segments.append({"text": sentence})
    # TTS using audio as source
    tts_model_str = tts_model
    start_time = time.time()
    target_audios = []
    tts_coroutines = []
    if tts_model_str  == "xtts":
        tts_model = sieve.function.get(f"sieve/xtts")
        for i, segment in enumerate(segments):
            tts = tts_model.push(
                segment["text"],
                source_audio,
                stability=speech_stability,
                similarity_boost=speech_similarity_boost
            )
            tts_coroutines.append(tts)
    elif tts_model_str == "elevenlabs":
        tts_model = sieve.function.get(f"sieve/elevenlabs_speech_synthesis")
        if len(voice_id) == 0:
            # clone voice
            cloning_model = sieve.function.get("sieve/elevenlabs_voice_cloning")
            voice_cloning = cloning_model.run(source_audio)
            print(voice_cloning)
            voice_id = voice_cloning["voice_id"]
        for i, segment in enumerate(segments):
            if voice_id and len(voice_id) > 0:
                tts = tts_model.push(
                    segment["text"],
                    voice_id=voice_id,
                    stability=speech_stability,
                    similarity_boost=speech_similarity_boost
                )
            else:
                tts = tts_model.push(
                    segment["text"],
                    stability=speech_stability,
                    similarity_boost=speech_similarity_boost
                )
                
            tts_coroutines.append(tts)
        
        if cleanup_voice_id:
            # delete voice
            cloning_model = sieve.function.get("sieve/elevenlabs_voice_cloning")
            cloning_model.run(source_audio, delete_voice_id=voice_id)
    elif tts_model_str == "playht":
        tts_model = sieve.function.get(f"sieve/playht_speech_synthesis")
        if len(voice_id) == 0:
            # clone voice
            cloning_model = sieve.function.get("sieve/playht_voice_cloning")
            voice_cloning = cloning_model.run(source_audio)
            print(voice_cloning)
            voice_id = voice_cloning["id"]
        for i, segment in enumerate(segments):
            if voice_id and len(voice_id) > 0:
                tts = tts_model.push(
                    segment["text"],
                    voice=voice_id,
                )
            else:
                tts = tts_model.push(
                    segment["text"],
                )
                
            tts_coroutines.append(tts)

        if cleanup_voice_id:
            # delete voice
            cloning_model = sieve.function.get("sieve/playht_voice_cloning")
            cloning_model.run(source_audio, delete_voice_id=voice_id)
    else:
        raise ValueError(f"Unsupported TTS model: {tts_model}. Please use one of the following: xtts, elevenlabs, playht")
    for tts in tts_coroutines:
        target_audios.append(tts.result())
    print(f"Time taken for TTS: {time.time() - start_time} seconds")
    
    if refine_target_audio:
        # Refine each target audio snippet
        start_time = time.time()
        refined_target_audios = []
        audio_enhancement_coroutines = []
        for target_audio in target_audios:
            enhanced_audio = sieve.function.get("sieve/audio_enhancement").push(target_audio, filter_type="all")
            audio_enhancement_coroutines.append(enhanced_audio)
        for i, enhanced_audio in enumerate(audio_enhancement_coroutines):
            try:
                refined_target_audios.append(enhanced_audio.result())
            except Exception as e:
                print(f"Exception at index {i} of audio_enhancement_coroutines: {e}, using target_audios[{i}] instead.")
                print(f"target audio is {target_audios[i]}")
                refined_target_audios.append(target_audios[i])
        print(f"Time taken to refine target audio: {time.time() - start_time} seconds")
        target_audios = refined_target_audios
    
    # Combine target audios with gaps
    from pydub import AudioSegment
    combined_audio = AudioSegment.empty()
    for i, target_audio in enumerate(target_audios):
        # Trim silence from target_audio
        start_time = time.time()
        target_audio_path = target_audio.path
        segment_audio = AudioSegment.from_wav(target_audio_path)
        trimmed_audio = trim_silence_from_audio_loaded(segment_audio)
        if i < len(segments) - 1:
            try:
                gap_duration = (segments[i+1]["start"] - segments[i]["words"][-1]["end"]) * 1000  # Convert to milliseconds
            except KeyError:
                print(f"KeyError at index {i} of segments. Using default gap duration of 0.05 seconds.")
                gap_duration = 0.05 * 1000
            gap = AudioSegment.silent(duration=gap_duration)
            combined_audio += trimmed_audio + gap
        else:
            combined_audio += trimmed_audio
    combined_audio.export("combined_audio.wav", format="wav")
    target_audio = sieve.Audio(path="combined_audio.wav")

    # Combine audio and video with Retalker
    start_time = time.time()
    retalker = sieve.function.get("sieve/video_retalking")
    output_video = retalker.run(source_video, target_audio, low_resolution=low_resolution, low_fps=low_fps)
    print(f"Time taken to combine audio and video: {time.time() - start_time} seconds")

    # Trim silence from output_video
    output_video_path = output_video.path
    if os.path.exists("output_video_trimmed.mp4"):
        os.remove("output_video_trimmed.mp4")
    start_time = time.time()
    trim_silence_from_video(output_video_path, "output_video_trimmed.mp4")
    print(f"Time taken to trim silence: {time.time() - start_time} seconds")

    trimmed_video = sieve.Video(path="output_video_trimmed.mp4")
    return trimmed_video



    
    