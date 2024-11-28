import sieve
from utils import (
    trim_silence_from_audio_loaded,
    convert_to_format,
    extract_audio_from_video
)

audio_metadata = sieve.Metadata(
    title="Dubbing",
    description="Translate any video or audio to several languages",
    tags=["Audio", "Video", "Lip-Syncing", "Translation", "Speech", "TTS", "Voice Cloning"],
    readme=open("README.md", "r").read(),
)


@sieve.function(
    name="dubbing",
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
    metadata=audio_metadata,
    environment_variables=[
        sieve.Env(name="ELEVEN_LABS_API_KEY", description="API key for ElevenLabs", default=""),
        sieve.Env(name="PLAYHT_API_KEY", description="API key for ElevenLabs", default=""),
        sieve.Env(name="PLAYHT_API_USER_ID", description="API user ID for ElevenLabs", default="")
    ],
)
def dubbing(
    source_file: sieve.File,
    target_language: str = "spanish",
    tts_model: str = "xtts",
    voice_id: str = "",
    cleanup_voice_id: bool = False,
    low_resolution: bool = False,
    low_fps: bool = False,
    enable_lipsyncing: bool = True,
):
    """
    :param source_file: An audio or video input file to dub.
    :param target_language: The language to which the audio will be translated. Default is "spanish".
    :param tts_model: The Text-to-Speech model to use. Supported models are "xtts", "elevenlabs", and "playht". "elevenlabs" or "playht" are recommended for better quality but requires an ElevenLabs API key.
    :param voice_id: The ID of the voice to use. If none are set, the voice will be cloned from the source audio and used. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
    :param cleanup_voice_id: Whether to delete the voice after use. This is only applicable if the `tts_model` is set to "elevenlabs" or "playht".
    :param low_resolution: Whether to reduce the resolution of an input video to half of the original on each axis. Significantly speeds up inference. Defaults to False. Only applicable for video inputs.
    :param low_fps: Whether to reduce the fps of an input video to half of the original. Significantly speeds up inference. Defaults to False. Only applicable for video inputs.
    :param enable_lipsyncing: Whether to enable lip-syncing on the original video to the dubbed audio. Defaults to True. Only applicable for video inputs. Otherwise, audio is returned.
    :return: An audio file dubbed in the target language.
    """
    import os
    import time

    source_file_path = source_file.path

    video_extensions = set(["mp4", "avi", "mkv", "flv", "mov", "wmv", "webm"])

    audio_extensions = set(["wav", "mp3", "ogg", "flac", "m4a"])

    source_path_extension = os.path.splitext(source_file_path)[1][1:].lower()

    is_video = False

    # Validate source file
    if source_path_extension in video_extensions:
        print("Source file detected as video")
        is_video = True
        source_video = sieve.Video(path=source_file_path)
        input_audio_path = "/tmp/input.wav"
        extract_audio_from_video(source_video.path, input_audio_path)
        source_audio = sieve.Audio(path=input_audio_path)
    elif source_path_extension in audio_extensions:
        print("Source file detected as audio")
        is_video = False
        source_audio = sieve.Audio(path=source_file_path)
    else:
        raise ValueError(
            f"Unsupported file extension: {source_path_extension}. Please use one of the following: {video_extensions.union(audio_extensions)}. This function supports video and audio files."
        )

    target_language = target_language.lower()

    source_audio = convert_to_format(source_audio, "source_audio.wav", "wav")

    # Refine source_audio
    start_time = time.time()
    source_audio = sieve.function.get("sieve/audio_enhancement").run(source_audio, filter_type="noise", enhance_speed_boost=True)
    print(f"Time taken to refine source audio: {time.time() - start_time} seconds")

    start_time = time.time()
    text_info = sieve.function.get("sieve/speech_transcriber").run(source_audio)
    out = list(text_info)
    segments = []
    for i in range(len(out)):
        segments.extend(out[i]["segments"])

    print(f"Time taken to get text info: {time.time() - start_time} seconds")
    language_code = out[0]["language_code"]

    try:
        source_language = LANGUAGE_CODE_MAP[language_code]
    except KeyError:
        raise ValueError(
            f"Unsupported language code: {language_code}. Please use one of the following: {list(LANGUAGE_CODE_MAP.keys())}"
        )

    # Translate text from english to another
    start_time = time.time()
    language_translator = sieve.function.get("sieve/seamless_text2text")
    translations = []
    translation_coroutines = []
    for segment in segments:
        translation = language_translator.push(
            segment["text"], source_language, target_language
        )
        translation_coroutines.append(translation)
    for translation in translation_coroutines:
        translations.append(translation.result())

    print(f"Time taken to translate text: {time.time() - start_time} seconds")

    concat_translations = " ".join(translations)

    # TTS using audio as source
    tts_model_str = tts_model
    start_time = time.time()
    target_audios = []
    tts_coroutines = []

    # TTS Hyperparameters, to autoconfigure later.
    speech_stability: float = 0.5
    speech_similarity_boost: float = 0.63

    if tts_model_str  == "xtts":
        tts_model = sieve.function.get(f"sieve/xtts")
        for i, segment in enumerate(segments):
            if target_language in INVERSE_LANGUAGE_CODE_MAP:
                language_code = INVERSE_LANGUAGE_CODE_MAP[target_language]
            else:   
                language_code = target_language
                print("Language code not found in map for language, using language code as is")
            tts = tts_model.push(
                translations[i],
                source_audio,
                stability=speech_stability,
                similarity_boost=speech_similarity_boost,
                language_code=language_code,
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
        if voice_id and len(voice_id) > 0:
            tts = tts_model.push(
                concat_translations,
                voice_id=voice_id,
                stability=speech_stability,
                similarity_boost=speech_similarity_boost
            )
        else:
            tts = tts_model.push(
                concat_translations,
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
                    translations[i],
                    voice=voice_id,
                )
            else:
                tts = tts_model.push(
                    translations[i],
                )
                
            tts_coroutines.append(tts)

        if cleanup_voice_id:
            # delete voice
            cloning_model = sieve.function.get("sieve/playht_voice_cloning")
            cloning_model.run(source_audio, delete_voice_id=voice_id)
    else:
        raise ValueError(f"Unsupported TTS model: {tts_model_str}. Please use one of the following: xtts, elevenlabs, playht")
    for tts in tts_coroutines:
        target_audios.append(tts.result())
    print(f"Time taken for TTS: {time.time() - start_time} seconds")

    # Combine target audios with gaps
    from pydub import AudioSegment

    combined_audio = AudioSegment.empty()
    for i, target_audio in enumerate(target_audios):
        # Trim silence from target_audio
        start_time = time.time()
        target_audio_path = target_audio.path
        # Convert the audio to wav if it is not a wav
        if not target_audio_path.endswith('.wav'):
            new_path = os.path.splitext(target_audio_path)[0] + '.wav'
            convert_to_format(target_audio, new_path, 'wav')
            target_audio_path = new_path
        segment_audio = AudioSegment.from_wav(target_audio_path)
        trimmed_audio = trim_silence_from_audio_loaded(segment_audio)
        if i < len(segments) - 1:
            try:
                gap_duration = (
                    segments[i + 1]["start"] - segments[i]["words"][-1]["end"]
                ) * 1000  # Convert to milliseconds
            except KeyError:
                print(
                    f"KeyError at index {i} of segments. Using default gap duration of 0.05 seconds."
                )
                gap_duration = 0
            gap = AudioSegment.silent(duration=gap_duration)
            combined_audio += trimmed_audio + gap
        else:
            combined_audio += trimmed_audio
    combined_audio.export("combined_audio.wav", format="wav")
    target_audio = sieve.Audio(path="combined_audio.wav")

    # Lip-syncing if enable_lipsyncing is True and the input is a video
    if is_video and enable_lipsyncing:
        print("Running Lip-syncing...")
        out_video = sieve.function.get("sieve/video_retalking").run(
            source_video,
            target_audio,
            low_resolution,
            low_fps,
        )
        
        return out_video

    return target_audio


LANGUAGE_CODE_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ar": "arabic",
    "hi": "hindi",
    "nl": "dutch",
    "sv": "swedish",
    "fi": "finnish",
    "da": "danish",
    "pl": "polish",
    "hu": "hungarian",
    "el": "greek",
    "tr": "turkish",
    "he": "hebrew",
    "id": "indonesian",
    "ms": "malay",
    "th": "thai",
    "vi": "vietnamese",
    "cs": "czech",
    "ro": "romanian",
    "uk": "ukrainian",
    "fa": "persian",
    "af": "afrikaans",
    "sw": "swahili",
    "no": "norwegian",
    "et": "estonian",
    "lt": "lithuanian",
    "lv": "latvian",
    "sl": "slovenian",
    "sk": "slovak",
    "hr": "croatian",
    "sr": "serbian",
    "mk": "macedonian",
    "bs": "bosnian",
    "sq": "albanian",
    "cy": "welsh",
    "ga": "irish",
    "mt": "maltese",
    "is": "icelandic",
    "tl": "filipino",
    "yo": "yoruba",
    "ig": "igbo",
    "ha": "hausa",
    "zu": "zulu",
    "xh": "xhosa",
    "st": "sesotho",
    "so": "somali",
    "am": "amharic",
    "ne": "nepali",
    "bn": "bengali",
    "pa": "punjabi",
    "gu": "gujarati",
    "or": "odia",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "si": "sinhala",
    "my": "burmese",
    "ka": "georgian",
    "hy": "armenian",
    "kk": "kazakh",
    "uz": "uzbek",
    "mn": "mongolian",
    "ky": "kyrgyz",
    "tg": "tajik",
    "tk": "turkmen",
    "ps": "pashto",
    "sd": "sindhi",
    "ur": "urdu",
    "yi": "yiddish",
    "la": "latin",
}

INVERSE_LANGUAGE_CODE_MAP = {v: k for k, v in LANGUAGE_CODE_MAP.items()}

