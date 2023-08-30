import sieve

model_metadata = sieve.Metadata(
    description="A Low Complexity Speech Enhancement Framework for Full-Band Audio (48kHz) using on Deep Filtering.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_noise_reduction/main.py",
    tags=["Audio"],
    readme=open("MODEL_README.md", "r").read(),
)

import sieve

@sieve.Model(
    name="deepfilternet_v2",
    gpu = True,
    python_packages=[
        "torch==1.9.0",
        "torchaudio==0.9.0",
        "deepfilternet"
    ],
    system_packages=["zip", "unzip"],
    run_commands=[
        "mkdir -p /root/.cache/DeepFilterNet",
        "wget -c https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip -P /root/.cache/DeepFilterNet",
        "unzip /root/.cache/DeepFilterNet/DeepFilterNet3.zip -d /root/.cache/DeepFilterNet",
        "ls -l /root/.cache/DeepFilterNet",
    ],
)
class DeepFilterNetV2:
    def __setup__(self):
        from df.enhance import enhance, init_df, load_audio, save_audio
        self.model, self.df_state, _ = init_df()

    def __predict__(self, audio: sieve.Audio) -> sieve.Audio:
        from df.enhance import enhance, init_df, load_audio, save_audio
        audio, _ = load_audio(audio.path, sr=self.df_state.sr())
        enhanced = enhance(self.model, self.df_state, audio)
        save_audio("enhanced.wav", enhanced, self.df_state.sr())
        return sieve.Audio(path="enhanced.wav")

wf_metadata = sieve.Metadata(
    title="Remove Audio Background Noise",
    description="Remove background noise from audio.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_noise_reduction/main.py",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/cover.png"
    ),
    tags=["Audio"],
    readme=open("README.md", "r").read(),
)

@sieve.workflow(name="audio_background_noise_removal", metadata=wf_metadata)
def audio_enhance(audio: sieve.Audio) -> sieve.Audio:
    """
    :param audio: A noisy audio input
    :return: Denoised audio
    """
    return DeepFilterNetV2()(audio)

if __name__ == "__main__":
    sieve.push(
        workflow="audio_noise_reduction",
        inputs={
            "audio": {
                "url": "https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/input.wav"
            }
        },
    )
