import sieve

metadata = sieve.Metadata(
    title="Demucs",
    description="Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment.",
    tags=["Audio", "Speech"],
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/demucs.jpeg"
    ),
    readme=open("README.md", "r").read(),
)


@sieve.function(
    name="demucs",
    system_packages=["ffmpeg","soundstretch"],
    python_packages=["git+https://github.com/facebookresearch/demucs#egg=demucs","torch==2.0.1","diffq"],
    metadata=metadata,
    gpu=sieve.gpu.L4(),
    run_commands = [
        #running models once to load them
        "python3 -m demucs --two-stems=vocals -n mdx_extra 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n htdemucs 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n htdemucs_ft 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n htdemucs_6s 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n hdemucs_mmi 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n mdx 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n mdx_q 'test.wav'",
        "python3 -m demucs --two-stems=vocals -n mdx_extra_q 'test.wav'",
        
    ],
    cuda_version="11.8.0"

    )
def audio_seperator(

    file: sieve.File,
    model : str = "htdemucs_ft",
    overlap: float = 0.25,
    shifts: int = 0,
    mp3: bool = False,
    ):
    """
    :param audio: Audio file to be separated
    :param model: The model to be used for audio separation. Default is "htdemucs_ft". Check the README for more information on the available models.
    :param overlap: option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine. It can probably be reduced to 0.1 to improve a bit speed.
    :param shifts: The number of shifts to be used. performs multiple predictions with random shifts (a.k.a the shift trick) of the input and average them. This makes prediction SHIFTS times slower. Default is 0
    :param mp3: If True, the audio will be saved as mp3. Default is False
    :return: The separated audio files
    """
    import os
    import demucs.separate
    import subprocess

    model = model.lower().strip()
    if model not in ["htdemucs","htdemucs_ft","htdemucs_6s","hdemucs_mmi","mdx","mdx_extra","mdx_q","mdx_extra_q"]:
        raise Exception("Invalid model selected. Please select from the available models")

    if shifts > 5 or shifts < 0:
        raise Exception("Invalid shift value. Please select a value between 0 and 5")

    if overlap > 1 or overlap < 0:
        raise Exception("Invalid overlap value. Please select a value between 0 and 1")
    

    audio_path = "input_audio" +".wav"

    if file.path.endswith(".mp4"):
        print("Extracting audio from video")
        try:
            subprocess.run(["ffmpeg", "-i", file.path, audio_path, "-y"])
        except Exception as e:
            raise Exception("Failed to extract audio from video. Make sure video has audio.")
    else:
        audio_path = file.path

    if not os.path.isfile(audio_path):
        raise Exception("Failed to extract audio!")

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    file_extension = 'mp3' if mp3 else 'wav'


    print("file name: ", file_name)
    
    if mp3:
        demucs.separate.main(["--mp3","--two-stems", "vocals","--overlap",str(overlap),"--shifts",str(shifts),"-j","16", "-n", model, audio_path])
    else:
        demucs.separate.main(["--two-stems", "vocals","--overlap", str(overlap),"--shifts",str(shifts),"-j","16", "-n", model, audio_path])

    vocals = sieve.Audio(path = os.getcwd() + f"/separated/{model}/{file_name}/vocals.{file_extension}")
    no_vocals = sieve.Audio(path = os.getcwd() + f"/separated/{model}/{file_name}/no_vocals.{file_extension}")

    return vocals, no_vocals


if __name__ == "__main__":
    file = sieve.File(path = "test.wav")
    output = audio_seperator(
        file,
        )
    print(output)

