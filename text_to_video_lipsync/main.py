import sieve
from typing import Literal

metadata = sieve.Metadata(
    # title="Text to Video Lipsync",
    description="Generate a video of a person speaking a given text using lipsyncing.",
    code_url="https://github.com/sieve-community/examples/blob/main/text_to_video_lipsync",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/side-profile-text.png"
    ),
    tags=["Audio", "Speech", "TTS", "Voice Cloning", "Video", "Lipsyncing"],
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
        "ffmpeg-python"
    ],
    metadata=metadata,
)
def do(
    source_video: sieve.File,
    text: str,
    lipsync_engine: Literal["musetalk", "video_retalking"] = "musetalk",
    voice_engine: Literal["elevenlabs-voice-cloning", "cartesia-voice-cloning"] = "elevenlabs-voice-cloning",
    downsample_video: bool = True,
    voice_stability: float = 0.5,
    voice_style: float = 0
):
    '''
    :param source_video: video to lip-sync
    :param text: text to speak
    :param lipsync_engine: Lipsync engine to use. Supported engines: "musetalk", "video_retalking".
    :param voice_engine: voice engine to use. Supported engines: "cartesia", "elevenlabs".
    :param downsample_video: Whether to downsample the video to 720p.
    :param voice_stability: Value between 0 and 1. Increasing variability can make speech more expressive with output varying between re-generations. It can also lead to instabilities.
    :param voice_style: Value between 0 and 1. High values are recommended if the style of the speech should be exaggerated compared to the original source audio. Higher values can lead to more instability in the generated speech. Setting this to 0.0 will greatly increase generation speed and is the default setting.
    :return: A generated video file
    '''
    import os
    import time
    from utils import extract_audio_from_video, trim_silence_from_video, trim_silence_from_audio_loaded, convert_to_format

    source_video_path = source_video.path
    # ensure fps of video is 60 fps or less
    import ffmpeg
    # get fps of video with ffmpeg
    probe = ffmpeg.probe(source_video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['r_frame_rate'])
    print(f"Input video FPS: {fps}")
    # if fps greater than 60, downsample the video to 60 fps
    if fps > 60:
        print("Downsampling video to 60 fps...")
        ffmpeg.input(source_video_path).output('output_60fps.mp4', r=60, vcodec='libx264', crf=23, acodec='aac').run(overwrite_output=True)
        source_video_path = "output_60fps.mp4"
    source_audio_path = "source_audio.wav"
    if os.path.exists(source_audio_path):
        os.remove(source_audio_path)
    
    tts_engine = sieve.function.get("sieve/tts")

    print(f"Extracting audio from video...")
    # Extract audio from source_video
    extract_audio_from_video(source_video_path, source_audio_path)
    source_audio = sieve.File(path=source_audio_path)

    print(f"Split text into sentences...")
    # split text into sentences by punctuation
    import re
    segments = []
    for sentence in re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text):
        segments.append({"text": sentence})
    
    print(f"Generating speech for each segment...")
    tts_coroutines = []
    for i, segment in enumerate(segments):
            tts = tts_engine.push(
                voice = voice_engine,
                text = segment["text"],
                reference_audio = source_audio,
                stability = voice_stability,
                style = voice_style
            )
            tts_coroutines.append(tts)

    print(f"Waiting for speech to generate...")
    target_audios = []
    for tts in tts_coroutines:
        target_audios.append(tts.result())
    
    print(f"Combining output speech with gaps...")
    # Combine target audios with gaps
    from pydub import AudioSegment
    combined_audio = AudioSegment.empty()
    # create temp directory with tempdir
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, target_audio in enumerate(target_audios):
            # Trim silence from target_audio
            target_audio_path = target_audio.path
            if target_audio_path.endswith(".wav"):
                segment_audio = AudioSegment.from_wav(target_audio_path)
            elif target_audio_path.endswith(".mp3"):
                segment_audio = AudioSegment.from_mp3(target_audio_path)
            else:
                raise ValueError(f"Unsupported audio format: {target_audio_path}. Please use .wav or .mp3")
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
    target_audio = sieve.File(path="combined_audio.wav")

    # Combine audio and video with Retalker
    print("Lipsyncing video...")
    start_time = time.time()
    lipsync = sieve.function.get("sieve/lipsync")
    output_video = lipsync.run(source_video, target_audio, backend=lipsync_engine, downsample=downsample_video)

    # Trim silence from output_video
    output_video_path = output_video.path
    if os.path.exists("output_video_trimmed.mp4"):
        os.remove("output_video_trimmed.mp4")
    trim_silence_from_video(output_video_path, "output_video_trimmed.mp4")
    trimmed_video = sieve.File(path="output_video_trimmed.mp4")

    print("Done!")
    return trimmed_video



    
    