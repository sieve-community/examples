import sieve
import time

valid_tasks = ["upsample", "noise", "all"]

metadata = sieve.Metadata(
    title="Enhance Audio",
    description="Remove background noise from audio and upsample it.",
    code_url="https://github.com/sieve-community/examples/tree/main/audio_enhancement",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/audio_noise_reduction/cover.png"
    ),
    tags=["Audio", "Speech", "Enhancement", "Showcase"],
    readme=open("README.md", "r").read(),
)


@sieve.function(name="audio_enhancement", metadata=metadata)
def enhance_audio(audio: sieve.Audio, filter_type: str = "all", speed_boost: bool = False, enhancement_steps: int = 50) -> sieve.Audio:
    '''
    :param audio: An audio input (mp3 and wav supported)
    :param filter_type: Task to perform, one of ["upsample", "noise", "all"]
    :param enhancement_steps: Number of enhancement steps applied to the audio between 10 and 150. Higher values may improve quality but will take longer to process. Defaults to 50. Only applicable if enhance_speed_boost is False.
    :param speed_boost: If True, use a faster, more experimental model for audio enhancement. Defaults to False.
    :return: enhanced + denoised audio
    '''
    audio_format = audio.path.split('.')[-1]
    if audio_format not in ['mp3', 'wav']:
        raise ValueError("Audio format must be mp3 or wav")

    task = filter_type.strip().lower()
    if task not in valid_tasks:
        raise ValueError(f"Task must be one of {valid_tasks}")

    if speed_boost:
        enhancement_model = "resemble-enhance"
    else:
        enhancement_model = "audiosr"
    
    if filter_type == "all":
        print("Running both noise reduction and upsampling")
    else:
        print(f"Running {filter_type} task")
    print("speed_boost:", speed_boost)
    print("enhancement_steps:", enhancement_steps)
    print("-"*50)

    enhance_func = sieve.function.get(f"sieve/{enhancement_model}")
    denoise_func = sieve.function.get("sieve/resemble-enhance")

    # use resemble enhance for both tasks if speed_boost is True
    if speed_boost and task == "all":
        # enhancement steps could be between 10 and 150, scale that to be between 1 and 100
        new_enhancement_steps = int(1 + (enhancement_steps - 10) * (99 / 140))
        print("Running joint enhancement and denoising task")
        duration = time.time()
        enhance_func = sieve.function.get("sieve/resemble-enhance")
        val = enhance_func.run(audio, process="enhance", cfm_func_evals=new_enhancement_steps)
        duration = time.time() - duration
        print(f"Audio enhanced and denoised in {duration} seconds")
        print("-"*50)
        return val

    if task == "upsample":
        print("Running upsampling task")
        duration = time.time()
        if enhancement_model == "audiosr":
            val = enhance_func.run(audio, enhancement_steps)
        else:
            val = enhance_func.run(audio, process="enhance", cfm_func_evals=100)
        duration = time.time() - duration
        print(f"Audio upsampled to 48kHz in {duration} seconds")
        print("-"*50)
        return val
    elif task == "noise":
        print("Running noise reduction task")
        duration = time.time()
        val = denoise_func.run(audio, process="denoise", cfm_func_evals=100)
        duration = time.time() - duration
        print(f"Audio denoised in {duration} seconds")
        print("-"*50)
        return val

    print("Running upsampling task")
    duration = time.time()
    if enhancement_model == "audiosr":
        enhanced = enhance_func.run(audio, enhancement_steps)
    else:
        enhanced = enhance_func.run(audio, process="enhance", cfm_func_evals=100)
    duration = time.time() - duration
    print(f"Audio upsampled to 48kHz in {duration} seconds")
    print("-"*50)

    print("Running noise reduction task")
    duration = time.time()
    denoised = denoise_func.run(enhanced, process="denoise", cfm_func_evals=100)
    duration = time.time() - duration
    print(f"Audio denoised in {duration} seconds")
    print("-"*50)
    
    return denoised
