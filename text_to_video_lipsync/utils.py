import sieve

def trim_audio_into_snippets(audio_path: str, timestamps: list) -> list:
    """
    Trims an audio file into different snippets based on given timestamps.

    Parameters:
    - audio_path: path to the input audio file.
    - timestamps: list of tuples, each containing start and end timestamps in seconds.

    Returns:
    - List of paths to each of the audio snippets.
    """
    from pydub import AudioSegment
    from pydub.utils import make_chunks

    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Initialize list to store paths to audio snippets
    snippet_paths = []

    # Loop over each timestamp pair and trim the audio
    for i, (start, end) in enumerate(timestamps):
        # Convert timestamps from seconds to milliseconds
        start_ms = start * 1000
        end_ms = end * 1000

        # Trim the audio
        snippet = audio[start_ms:end_ms]

        # Save the snippet to a new file
        snippet_path = f"snippet_{i}.wav"
        snippet.export(snippet_path, format="wav")

        # Add the snippet path to the list
        snippet_paths.append(snippet_path)

    return snippet_paths

def convert_to_format(audio: sieve.Audio, output_path: str, output_format: str = "mp3") -> sieve.Audio:
    """
    Converts a video or audio file to an mp3 or wav file.

    Parameters:
    - input_path: path to the input video or audio file.
    - output_path: path to save the output audio file.
    - output_format: format of the output audio file. Default is "mp3".

    Returns:
    - new sieve.Audio
    """
    from pydub import AudioSegment
    import os

    input_path = audio.path

    if output_format not in ["mp3", "wav"]:
        raise ValueError(f"Unsupported output format: {output_format}. Please use one of the following: mp3, wav")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    if os.path.splitext(input_path)[1] == ".mp3" and output_format == "mp3":
        return audio
    elif os.path.splitext(input_path)[1] == ".wav" and output_format == "wav":
        return audio

    # Load the input file
    audio_input = AudioSegment.from_file(input_path)

    # Export the audio in the desired format
    audio_input.export(output_path, format=output_format)

    return sieve.Audio(path=output_path)

def match_audio_length(source_audio_path: str, target_audio_path: str, output_path: str = "matched_audio.wav") -> str:
    """
    Stretches or shrinks the source audio to match the length of the target audio.

    Parameters:
    - source_audio_path: path to the source audio file.
    - target_audio_path: path to the target audio file.
    - output_path: path to save the matched audio. Default is "matched_audio.wav".

    Returns:
    - Path to the matched audio file.
    """
    import librosa
    import pyrubberband as pyrb
    import soundfile as sf

    # Load the audio files
    source_audio, sr_source = librosa.load(source_audio_path, sr=None)
    target_audio, sr_target = librosa.load(target_audio_path, sr=None)

    print(f"Source audio length: {len(source_audio) / sr_source} seconds")
    print(f"Target audio length: {len(target_audio) / sr_target} seconds")

    # Calculate the speed factor
    speed_factor = len(target_audio) / len(source_audio)

    print(f"Speed factor: {speed_factor}")

    # Stretch or shrink the source audio
    matched_audio = pyrb.time_stretch(source_audio, sr_source, 1/speed_factor)

    # Export the matched audio
    sf.write(output_path, matched_audio, sr_source, format='wav')

    return output_path

def match_audio_length_from_segment(source_audio, target_length_ms: int):
    """
    Stretches or shrinks the source audio to match the target length.

    Parameters:
    - source_audio: AudioSegment of the source audio file.
    - target_length_ms: target length in milliseconds.

    Returns:
    - AudioSegment of the matched audio.
    """
    import librosa
    import pyrubberband as pyrb
    import numpy as np
    from pydub import AudioSegment

    # Convert source_audio to numpy array
    source_audio_array = np.array(source_audio.get_array_of_samples())

    source_audio_length_s = len(source_audio_array) / source_audio.frame_rate
    target_audio_length_s = target_length_ms / 1000

    print(f"Source audio length: {source_audio_length_s} seconds")
    print(f"Target audio length: {target_audio_length_s} seconds")

    # Calculate the speed factor
    speed_factor = target_audio_length_s / source_audio_length_s

    print(f"Speed factor: {speed_factor}")

    # Stretch or shrink the source audio
    matched_audio_array = pyrb.time_stretch(source_audio_array, source_audio.frame_rate, 1/speed_factor)

    # Convert matched_audio_array back to AudioSegment
    matched_audio = AudioSegment(
        matched_audio_array.tobytes(),
        frame_rate=source_audio.frame_rate,
        sample_width=source_audio.sample_width,
        channels=source_audio.channels
    )

    return matched_audio

def speed_up_audio(audio_path: str, speed: float = 1.25, output_path: str = "sped_up_audio.wav") -> str:
    """
    Speeds up an audio file by a given factor without changing the pitch.

    Parameters:
    - audio_path: path to the input audio file.
    - speed: factor to speed up the audio by. Default is 1.25.
    - output_path: path to save the sped-up audio. Default is "sped_up_audio.wav".

    Returns:
    - Path to the sped-up audio file.
    """
    from pydub import AudioSegment
    from pydub.playback import play

    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Speed up the audio
    sped_up_audio = audio.speedup(playback_speed=speed)

    # Export the sped-up audio
    sped_up_audio.export(output_path, format="wav")

    return output_path

def trim_audio_into_snippets(audio_path: str, timestamps: list) -> list:
    """
    Trims an audio file into different snippets based on given timestamps.

    Parameters:
    - audio_path: path to the input audio file.
    - timestamps: list of tuples, each containing start and end timestamps in seconds.

    Returns:
    - List of paths to each of the audio snippets.
    """
    from pydub import AudioSegment
    from pydub.utils import make_chunks

    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Initialize list to store paths to audio snippets
    snippet_paths = []

    # Loop over each timestamp pair and trim the audio
    for i, (start, end) in enumerate(timestamps):
        # Convert timestamps from seconds to milliseconds
        start_ms = start * 1000
        end_ms = end * 1000

        # Trim the audio
        snippet = audio[start_ms:end_ms]

        # Save the snippet to a new file
        snippet_path = f"snippet_{i}.wav"
        snippet.export(snippet_path, format="wav")

        # Add the snippet path to the list
        snippet_paths.append(snippet_path)

    return snippet_paths


def extract_audio_from_video(video_path: str, output_path: str):
    """
    Extracts audio from a video and saves it to an output path.

    Parameters:
    - video_path: path to the input video file.
    - output_path: path to save the extracted audio.
    """
    from moviepy.editor import VideoFileClip

    # Load the video file
    clip = VideoFileClip(video_path)

    # Extract audio
    audio = clip.audio

    # Save audio to output path
    audio.write_audiofile(output_path)


def trim_silence_from_video(video_path, output_path, silence_thresh=-50.0):
    """
    Trims silence from the beginning and end of a video.

    Parameters:
    - video_path: path to the input video file.
    - output_path: path to save the trimmed video.
    - silence_thresh: threshold in dB. Anything quieter than this will be considered silence.
    - chunk_size: how long to analyze sound for (in ms).
    """

    from moviepy.editor import VideoFileClip
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    # Load the video file
    clip = VideoFileClip(video_path)

    # Convert video audio to pydub's AudioSegment format
    audio = AudioSegment.from_file(video_path, codec="aac")

    # Detect non-silent chunks
    non_silence_ranges = detect_nonsilent(audio, min_silence_len=1, silence_thresh=silence_thresh)

    # If there are non-silent chunks, trim the video file based on the first and last non-silent chunk
    if non_silence_ranges:
        start_trim = max(0, non_silence_ranges[0][0] - 250)  # Add a quarter second buffer to the start, if possible
        end_trim = min(len(audio), non_silence_ranges[-1][1] + 250)  # Add a quarter second buffer to the end, if possible
        trimmed_clip = clip.subclip(start_trim / 1000.0, end_trim / 1000.0)  # Convert ms to seconds
        trimmed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    else:
        print("All audio is silent. Nothing to trim!")

def trim_silence_from_audio(audio_path, output_path, silence_thresh=-50.0):
    """
    Trims silence from the beginning and end of an audio.

    Parameters:
    - audio_path: path to the input audio file.
    - output_path: path to save the trimmed audio.
    - silence_thresh: threshold in dB. Anything quieter than this will be considered silence.
    """

    from pydub import AudioSegment

    # Load the audio file
    audio = AudioSegment.from_file(audio_path, format="wav")

    # Detect non-silent chunks
    out = trim_silence_from_audio_loaded(audio, silence_thresh=silence_thresh)

    # Export the trimmed audio
    out.export(output_path, format="wav")

def trim_silence_from_audio_loaded(audio, silence_thresh=-50.0, buffer_ms=250):
    from pydub.silence import detect_nonsilent
    # Detect non-silent chunks
    non_silence_ranges = detect_nonsilent(audio, min_silence_len=1, silence_thresh=silence_thresh)

    # If there are non-silent chunks, trim the audio file based on the first and last non-silent chunk
    if non_silence_ranges:
        start_trim = max(0, non_silence_ranges[0][0] - buffer_ms)  # Add a quarter second buffer to the start, if possible
        end_trim = min(len(audio), non_silence_ranges[-1][1] + buffer_ms)  # Add a quarter second buffer to the end, if possible
        trimmed_audio = audio[start_trim:end_trim]
        return trimmed_audio
    else:
        print("All audio is silent. Nothing to trim!")
        return audio