import sieve

@sieve.Model(
    name="fullsubnet",
    gpu = True,
    python_version="3.8",
    python_packages=[
       "torch==1.12.1",
        "torchvision==0.13.1",
        "torchaudio==0.12.1",
        "librosa==0.9.2",
        "joblib==1.1.0",
        "pesq==0.0.3",
        "pypesq==1.2.4",
        "pystoi==0.3.3",
        "tqdm==4.62.3",
        "toml==0.10.2",
        "colorful==0.5.4",
        "torch_complex==0.2.1",
    ],
    system_packages=[
        "libsndfile1-dev",
        "ffmpeg",
    ],
    run_commands=[
        "mkdir -p /root/.cache/audio_enhance/models",
        "wget -c 'https://storage.googleapis.com/sieve-public-model-assets/fullsubnet/best_model.tar' -P /root/.cache/audio_enhance/models/"
    ],
)
class FullSubNet():
    def __setup__(self):
        from speech_enhance.tools.denoise_hf_clone_voice import start

    def __predict__(self, audio: sieve.Audio) -> sieve.Audio:
        '''
        :param audio: A noisy audio input
        :return: Denoised audio
        '''
        from speech_enhance.tools.denoise_hf_clone_voice import start
        result = start(to_list_files=[audio.path])
        return sieve.Audio(path=result[0])


@sieve.workflow(name="audio_noise_reduction")
def audio_enhance(audio: sieve.Audio) -> sieve.Audio:
    '''
    :param audio: A noisy audio input
    :return: Denoised audio
    '''
    return FullSubNet()(audio)