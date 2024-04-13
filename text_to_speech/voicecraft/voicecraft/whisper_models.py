from voicecraft.utils import get_transcribe_state, get_random_string


class WhisperxAlignModel:
    def __init__(self,device):
        from whisperx import load_align_model
        self.device = device
        self.model, self.metadata = load_align_model(language_code="en", device=device)

    def align(self, segments, audio_path):
        from whisperx import align, load_audio
        audio = load_audio(audio_path)
        return align(segments, self.model, self.metadata, audio, self.device, return_char_alignments=False)["segments"]


class WhisperModel:
    def __init__(self, model_name, device):
        from whisper import load_model
        self.model = load_model(model_name, device)

        from whisper.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(multilingual=False)
        self.supress_tokens = [-1] + [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path, suppress_tokens=self.supress_tokens, word_timestamps=True)["segments"]


class WhisperxModel:
    def __init__(self, model_name, align_model: WhisperxAlignModel,device):
        from whisperx import load_model
        self.model = load_model(model_name, device, asr_options={"suppress_numerals": True, "max_new_tokens": None, "clip_timestamps": None, "hallucination_silence_threshold": None})
        self.align_model = align_model

    def transcribe(self, audio_path):
        segments = self.model.transcribe(audio_path, batch_size=8)["segments"]
        return self.align_model.align(segments, audio_path)



def load_models(device, whisper_backend_name="whisper", whisper_model_name="base.en", alignment_model_name="whisperx", voicecraft_model_name = "giga830M"):
    import os
    import torch
    from voicecraft.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    )
    from voicecraft.models import voicecraft
    
    
    if voicecraft_model_name == "giga330M_TTSEnhanced":
        voicecraft_model_name = "gigaHalfLibri330M_TTSEnhanced_max16s"

    print("LOADING ALIGNMENT MODELS")
    if alignment_model_name is not None:
        align_model = WhisperxAlignModel(device)
    print("done ALIGNMENT MODELS")

    print("load transcribe model")
    if whisper_model_name is not None:
        if whisper_backend_name == "whisper":
            transcribe_model = WhisperModel(whisper_model_name,device)
        else:
            if align_model is None:
                raise print("Align model required for whisperx backend")
            transcribe_model = WhisperxModel(whisper_model_name, align_model,device)
    print("done load transcribe model")

    voicecraft_name = f"{voicecraft_model_name}.pth"
    ckpt_fn = f"/root/.cache/pretrained_models/{voicecraft_name}"
    encodec_fn = "/root/.cache/pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(ckpt_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true /root/.cache/pretrained_models/{voicecraft_name}")
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th /root/.cache/pretrained_models/encodec_4cb2048_giga.th")


    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    voicecraft_model = {
        "ckpt": ckpt,
        "model": model,
        "text_tokenizer": TextTokenizer(backend="espeak"),
        "audio_tokenizer": AudioTokenizer(signature=encodec_fn)
    }

    return transcribe_model, align_model, voicecraft_model


def align_segments(TMP_PATH,transcript, audio_path):
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
    import json
    import os
    config_string = 'task_language=eng|os_task_file_format=json|is_text_type=plain'
    tmp_transcript_path = os.path.join(TMP_PATH, f"{get_random_string()}.txt")
    tmp_sync_map_path = os.path.join(TMP_PATH, f"{get_random_string()}.json")
    with open(tmp_transcript_path, "w") as f:
        f.write(transcript)

    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(tmp_transcript_path)
    task.sync_map_file_path_absolute = os.path.abspath(tmp_sync_map_path)
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    with open(tmp_sync_map_path, "r") as f:
        return json.load(f)
    

def align(transcript, audio_path,TMP_PATH, align_model):
    if align_model is None:
        raise Exception("Align model not loaded")
    fragments = align_segments(TMP_PATH,transcript, audio_path)
    segments = [{
        "start": float(fragment["begin"]),
        "end": float(fragment["end"]),
        "text": " ".join(fragment["lines"])
    } for fragment in fragments["fragments"]]
    segments = align_model.align(segments, audio_path)
    state = get_transcribe_state(segments)

    return state

def transcribe(audio_path, transcribe_model):
    if transcribe_model is None:
        raise Exception("Transcribe model not loaded")
    
    segments = transcribe_model.transcribe(audio_path)
    state = get_transcribe_state(segments)

    return state