import sieve


class AudioUpscaler:
    def __init__(self) -> None:
        import audioread
        import numpy as np
        import torch
        from hifi_gan_bwe import BandwidthExtender
        import soundfile
        from tqdm import tqdm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BandwidthExtender.from_pretrained(
            "hifi-gan-bwe-10-42890e3-vctk-48kHz"
        ).to(self.device)

    def upscale(self, audio_path, output_path):
        import audioread
        import numpy as np
        import torch
        from hifi_gan_bwe import BandwidthExtender
        import soundfile
        from tqdm import tqdm

        with audioread.audio_open(audio_path) as input_:
            sample_rate = input_.samplerate
            x = (
                np.hstack([np.frombuffer(b, dtype=np.int16) for b in input_])
                .reshape([-1, input_.channels])
                .astype(np.float32)
                / 32767.0
            )

        with torch.no_grad():
            y = (
                torch.stack(
                    [
                        self.model(torch.from_numpy(x).to(self.device), sample_rate)
                        for x in x.T
                    ]
                )
                .T.cpu()
                .numpy()
            )

        # save the output file
        soundfile.write(output_path, y, samplerate=int(self.model.sample_rate))

        return output_path


model_metadata = sieve.Metadata(
    description='Unofficial implementation of HiFi-GAN+ from the paper "Bandwidth Extension is All You Need" by Su, et al.',
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhance/hifigan.py",
    tags=["Audio"],
    readme=open("HIFI_README.md", "r").read(),
)


@sieve.Model(
    name="hifi_gan_plus",
    gpu=True,
    machine_type="a100-20gb",
    python_version="3.9",
    cuda_version="11.8",
    python_packages=["hifi-gan-bwe==0.1.14"],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg",
        "libavcodec58",
        "libsndfile1",
    ],
    run_commands=[
        "python -c 'from hifi_gan_bwe import BandwidthExtender; model = BandwidthExtender.from_pretrained(\"hifi-gan-bwe-10-42890e3-vctk-48kHz\")'"
    ],
    metadata=model_metadata,
)
class HiFiGanPlus:
    def __setup__(self):
        self.audio_upscaler = AudioUpscaler()

    def __predict__(self, audio: sieve.Audio) -> sieve.Audio:
        import os
        import subprocess

        if os.path.exists("out.wav"):
            os.remove("out.wav")

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
                "20",
                "-c",
                "copy",
                "temp_audio/out%04d.wav",
            ]
        )

        out_list = []
        # process each chunk separately
        for file in sorted(os.listdir("temp_audio")):
            print(f"Processing {file}")
            out = self.audio_upscaler.upscale(f"temp_audio/{file}", f"output/{file}")
            out_list.append(out)

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
                "out.wav",
            ]
        )

        # remove temp_audio directory
        subprocess.run(["rm", "-rf", "temp_audio"])

        # remove output directory
        subprocess.run(["rm", "-rf", "output"])

        return sieve.Audio(path="out.wav")
