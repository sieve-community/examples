import sieve

@sieve.Model(
    name="audiogen",
    python_version="3.9",
    cuda_version="11.8",
    gpu=sieve.gpu.L4(),
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_packages=[
        "torch==2.1.0",
        "git+https://github.com/facebookresearch/audiocraft.git"
    ]
)
class AudioGen:
    def __setup__(self):
        from audiocraft.models import AudioGen

        self.model = AudioGen.get_pretrained('facebook/audiogen-medium')

    def __predict__(self, prompt: str, duration: float = 5.0) -> sieve.File:
        from audiocraft.data.audio import audio_write

        self.model.set_generation_params(duration=duration)
        descriptions = [prompt]
        waveforms = self.model.generate(descriptions)
        waveform = waveforms[0]

        audio_path = f'/tmp/audio'
        audio_write(audio_path, waveform.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        return sieve.File(path='/tmp/audio.wav')

