import sieve

@sieve.function(
    name="video-sound-effect",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
)
def video_sound_effect(video: sieve.File, duration: float = 5.0) -> sieve.File:
    if duration > 10.0:
        raise ValueError("Duration must be less than 10 seconds")
    if duration < 0.0:
        raise ValueError("Duration must be greater than 0 seconds")

    # ffmpeg get video length
    import subprocess
    command = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video.path}'
    video_length = float(subprocess.check_output(command, shell=True).decode('utf-8').strip())

    if video_length < duration:
        raise ValueError("Video length must be greater than duration")
    
    # cut video to duration with overwrite option
    output_video_path = f'/tmp/cut_video.mp4'
    command = f'ffmpeg -y -i {video.path} -t {duration} -c copy {output_video_path}'
    subprocess.run(command, shell=True)
    
    # get middle frame of video at half of the duration
    middle_time = duration / 2
    command = f'ffmpeg -ss {middle_time} -i {output_video_path} -vframes 1 -f image2 -y /tmp/frame.png'
    subprocess.run(command, shell=True)

    cogvlm = sieve.function.get("sieve/cogvlm-chat")
    prompt = "describe what you might hear in this video in detail."
    description = cogvlm.run(sieve.Image(path="/tmp/frame.png"), prompt, vqa_mode=True)

    audiogen = sieve.function.get("sieve-internal/audiogen")
    sound = audiogen.run(description, duration)
    sound_path = sound.path

    # remove old sound and put sound effect to video
    final_video_path = f'/tmp/final_video.mp4'
    # check if video has audio
    command = f'ffprobe -i {output_video_path} -show_streams -select_streams a:0 -loglevel error'
    has_audio = subprocess.run(command, shell=True).returncode == 0
    # remove audio
    if has_audio:
        new_video_path = f'/tmp/no_audio_video.mp4'
        command = f'ffmpeg -y -i {output_video_path} -c copy -an {new_video_path}'
        subprocess.run(command, shell=True)
    else:
        new_video_path = output_video_path
    # merge video and sound with the final length based on whichever is shorter, and encode video in h264 for browser compatibility
    command = f'ffmpeg -y -i {new_video_path} -i {sound_path} -filter_complex "[1:0] apad" -shortest -c:v libx264 -c:a aac {final_video_path}'
    subprocess.run(command, shell=True)
    
    return sieve.File(path=final_video_path)

if __name__ == "__main__":
    video = sieve.File(path="/Users/Mokshith/Documents/Sieve/experiments/examples/tests/autocrop/short-tennis.mp4")
    duration = 5.0
    print(video_sound_effect(video, duration))