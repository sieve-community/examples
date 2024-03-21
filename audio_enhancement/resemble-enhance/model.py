import sieve

device = "cuda"
solver = "midpoint"

metadata = sieve.Metadata(
    description="Resemble Enhance is an AI-powered tool that aims to improve the overall quality of speech by performing denoising and enhancement",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement/resemble-enhance",
    image = sieve.Image(
        path = "enhancer_logo.png",
    ),
    tags = ["Audio", "Speech", "Denoising", "Enhancement"],
    readme = open("README.md", "r").read(),
)

@sieve.Model(
        name="resemble-enhance",
        gpu="l4",
        python_version="3.11",
        cuda_version="11.8",
        system_packages=["ffmpeg", "libsndfile1", "git-lfs"],
        metadata=metadata,
        run_commands=[
            "pip install resemble-enhance",
            "mkdir -p /root/.cache/resemble-enhance/ds/G/default",
            "wget -O /root/.cache/resemble-enhance/hparams.yaml https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/hparams.yaml?download=true",
            "wget -O /root/.cache/resemble-enhance/ds/G/latest https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/latest?download=true",
            "wget -O /root/.cache/resemble-enhance/ds/G/default/mp_rank_00_model_states.pt https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt?download=true",
        ],
)
class Model:
    def __setup__(self):
        from resemble_enhance.enhancer.inference import load_enhancer

        self.denoise = load_enhancer(None, device)

    def __predict__(self, file: sieve.Audio, process: str = "denoise", cfm_func_evals: int = 64, cfm_temp: float = 0.5):
        """
        :param file: audio to process
        :param process: Options: "denoise", "enhance", or "all" - Type of processing to apply to audio. 
        :param cfm_func_evals: Range: (0, 100) - Number of function evaluations to use for the CFM solver (Higher values in general yield better results but can be slower)
        :param cfm_temp: Range: (0, 1) - Temperature to use for the CFM solver (Higher values in general yield better results but can reduce stability)
        :return: Processed audio
        """
        from resemble_enhance.enhancer.inference import denoise, enhance
        import torchaudio
        import torch
        
        if process.lower() not in ["denoise", "enhance", "all"]:
            raise ValueError(f"Invalid process type: '{process}'. Expected 'denoise', 'enhance', or 'all'.")
        if cfm_func_evals < 1 or cfm_func_evals > 100:
            raise ValueError(f"Invalid number of function evaluations: '{cfm_func_evals}'. Expected value between 1 and 100.")
        if cfm_temp < 0 or cfm_temp > 1:
            raise ValueError(f"Invalid temperature: '{cfm_temp}'. Expected value between 0 and 1.")
        
        # Load audio
        input_path = file.path
        audio_waveform, sample_rate = torchaudio.load(input_path)
        audio_waveform = audio_waveform.mean(dim=0)

        # Set parameters based on process type
        lambda_value = 0.1 if process.lower() == "enhance" else 0.9 if process.lower() == "all" else None
        audio_processed, new_sample_rate = None, None

        # Process audio based on type
        if process.lower() in ["denoise"]:
            audio_processed, new_sample_rate = denoise(audio_waveform, sample_rate, device)
        if process.lower() in ["enhance", "all"]:
            audio_processed, new_sample_rate = enhance(audio_waveform, sample_rate, device, lambd=lambda_value, solver=solver, nfe=cfm_func_evals, tau=cfm_temp)

        # Save and return processed audio
        processed_audio_path = "processed_denoised.wav"
        torchaudio.save(processed_audio_path, audio_processed.clone().detach().unsqueeze(0), new_sample_rate)
        return sieve.Audio(path=processed_audio_path)