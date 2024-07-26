import sieve
from typing import Literal

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
    model : Literal["htdemucs","htdemucs_ft","htdemucs_6s","hdemucs_mmi","mdx","mdx_extra","mdx_q","mdx_extra_q"] = "htdemucs_ft",
    two_stems: Literal["None","vocals", "drums", "bass", "other", "guitar","piano"] = "None",
    overlap: float = 0.25,
    shifts: int = 0,
    audio_format: Literal["wav","mp3", "flac"] = "wav",
    ):
    """
    :param audio: Audio file to be separated
    :param model: The model to be used for audio separation. Default is "htdemucs_ft". Check the README for more information on the available models.
    :param two_stems: Only seperate audio into stem and no_stem. the stem which is specified will be seperated from the rest of the stems.
    :param overlap: option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine. It can probably be reduced to 0.1 to improve a bit speed.
    :param shifts: The number of shifts to be used. performs multiple predictions with random shifts (a.k.a the shift trick) of the input and average them. This makes prediction SHIFTS times slower. Default is 0
    :param audio_format: The format of the audio file to be returned. Default is "wav". You can choose from "mp3", "flac" or "wav".
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

    if two_stems in ["guiatar","piano"] and model != "htdemucs_6s":
        raise Exception("Stems guitar and piano are only available for model htdemucs_6s")

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

    command = ["--overlap", str(overlap), "--shifts", str(shifts), "-j", "16", "-n", model, audio_path]
    if audio_format != "wav":
        command.insert(0,f"--{audio_format}")
    if two_stems != "None":
        command.insert(0, "--two-stems")
        command.insert(1, two_stems)
    
    demucs.separate.main(command)
    dir_path = f"separated/{model}/{file_name}/"

    if two_stems == "None":
        audios = [
            sieve.File(path= f"{dir_path}vocals.{audio_format}"),
            sieve.File(path= f"{dir_path}drums.{audio_format}"),
            sieve.File(path= f"{dir_path}bass.{audio_format}"),
            sieve.File(path= f"{dir_path}other.{audio_format}")
        ]
        if model == "htdemucs_6s":
            audios.extend([
                sieve.File(path= f"{dir_path}guitar.{audio_format}"),
                sieve.File(path= f"{dir_path}piano.{audio_format}")
            ])
    else:
        audios = [
            sieve.File(path= f"{dir_path}{two_stems}.{audio_format}"),
            sieve.File(path= f"{dir_path}no_{two_stems}.{audio_format}")
        ]

    return audios




if __name__ == "__main__":
    file = sieve.File(path = "test.wav")
    output = audio_seperator(
        file,
        )
    print(output)
