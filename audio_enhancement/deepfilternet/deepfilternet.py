import sieve

model_metadata = sieve.Metadata(
    description="A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) using on Deep Filtering.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement/deepfilternet",
    tags=["Audio", "Speech", "Enhancement"],
    image=sieve.Image(
        url="https://everydayseries.com/content/images/2023/06/DeepFilterNet2-architecture.png"
    ),
    readme=open("DEEPFILTER_README.md", "r").read(),
)


@sieve.Model(
    name="deepfilternet_v2",
    gpu=True,
    python_packages=["torch==1.9.0", "torchaudio==0.9.0", "deepfilternet"],
    system_packages=["zip", "unzip", "ffmpeg"],
    run_commands=[
        "mkdir -p /root/.cache/DeepFilterNet",
        "wget -c https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip -P /root/.cache/DeepFilterNet",
        "unzip /root/.cache/DeepFilterNet/DeepFilterNet3.zip -d /root/.cache/DeepFilterNet",
        "ls -l /root/.cache/DeepFilterNet",
    ],
    metadata=model_metadata,
)
class DeepFilterNetV2:
    def __setup__(self):
        from df.enhance import enhance, init_df, load_audio, save_audio

        self.model, self.df_state, _ = init_df()

    def __predict__(self, audio: sieve.Audio) -> sieve.Audio:
        """
        :param audio: audio to remove background noise from
        :return: audio with background noise removed
        """
        from df.enhance import enhance, init_df, load_audio, save_audio

        audio_path_ending = audio.path.split(".")[-1].strip()
        if audio_path_ending not in ["wav", "mp3", "flac"]:
            audio_path_ending = "wav"

        import os
        import subprocess

        if os.path.exists("out.wav"):
            os.remove(f"out.wav")

        if os.path.exists("out.mp3"):
            os.remove(f"out.mp3")

        if os.path.exists("out.flac"):
            os.remove(f"out.flac")

        # split audio in 10s chunks and process each chunk separately
        # create temp directory in temp_audio
        if not os.path.exists("temp_audio"):
            os.mkdir("temp_audio")

        # make output directory
        if not os.path.exists("output"):
            os.mkdir("output")

        # split audio in 20s chunks
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio.path,
                "-f",
                "segment",
                "-segment_time",
                "180",
                "-c",
                "copy",
                f"temp_audio/out%04d.{audio_path_ending}",
            ]
        )

        out_list = []
        # process each chunk separately
        for file in sorted(os.listdir("temp_audio")):
            print(f"Processing {file}")
            audio, _ = load_audio(f"temp_audio/{file}", sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
            save_audio(f"output/{file}", enhanced, self.df_state.sr())
            out_list.append(f"output/{file}")

        # write filenames to filelist.txt
        with open("filelist.txt", "w") as f:
            for file in out_list:
                f.write(f"file '{file}'\n")

        # concatenate files
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                "filelist.txt",
                "-c",
                "copy",
                f"out.{audio_path_ending}",
            ]
        )

        # remove temp_audio directory
        subprocess.run(["rm", "-rf", "temp_audio"])

        # remove output directory
        subprocess.run(["rm", "-rf", "output"])

        return sieve.Audio(path=f"out.{audio_path_ending}")
