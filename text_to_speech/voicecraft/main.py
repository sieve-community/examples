import sieve

metadata = sieve.Metadata(
    description="Generate audio similar to speaker or edit real audio with VoiceCraft",
    tags=["Audio", "Speech", "TTS"],
    image=sieve.Image(
        url="https://the-decoder.com/wp-content/uploads/2024/04/AI_voice_cloning_illustration.jpeg"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="voicecraft",
    metadata=metadata,
    gpu=sieve.gpu.L4(), 
    python_packages=[
        "numpy",
        "tensorboard",
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "xformers==0.0.22",
        "phonemizer",
        "datasets",
        "torchmetrics",
        "openai-whisper>=20231117",
        "nltk>=3.8.1",
        "whisperx>=3.1.1",
        "pydantic",
        "spacy"
    ],

    run_commands=[

        "pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft",
        "pip install aeneas>=1.7.3.0",
        "pip uninstall pydantic --y",
        "pip uninstall spacy --y",
        "pip install pydantic",
        "pip install spacy",

        "mkdir -p /root/.cache/torch/hub/checkpoints",
        "mkdir temp",
        "mkdir /root/.cache/pretrained_models/",
        "mkdir /root/.cache/whisper/",
        "wget https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth -q -P /root/.cache/torch/hub/checkpoints", #downloading the alignment model
        "wget https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt -q -P /root/.cache/whisper", # downloading whisper model
        "wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/giga830M.pth\?download\=true",
        "mv giga830M.pth\?download\=true /root/.cache/pretrained_models/giga830M.pth",
        "wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th",
        "mv encodec_4cb2048_giga.th /root/.cache/pretrained_models/encodec_4cb2048_giga.th"

       
    ],
    system_packages=[
        "ffmpeg",
        "espeak",
        "python3-dev",
        "espeak-data",
        "libespeak1",
        "libespeak-dev",
        "festival*",
        "build-essential",
        "flac",
        "libasound2-dev",
        "libsndfile1-dev",
        "vorbis-tools",
        "libxml2-dev",
        "libxslt-dev",
        "zlib1g-dev",
        "python3-dev",
        
    ],
    python_version="3.10.12",
    cuda_version="11.8.0"
)
class VoiceCraftTTS:
    def __setup__(self):
        import os
        import torch
        from voicecraft.whisper_models import load_models


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ['USER'] = 'seieve_fn'
        self.TMP_PATH = 'temp/'

        if not os.path.exists(self.TMP_PATH):
          os.makedirs(self.TMP_PATH)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcribe_model, self.align_model, self.voicecraft_model = load_models(self.device)

    def __predict__(
            self,
            reference_audio: sieve.Audio,
            input_text: str, 
            mode: str = "TTS",
            prompt_end_time: float  = -1.0, 
            edit_start_time: float = -1.0, 
            edit_end_time: float = -1.0, 
            split_text: str = "Newline", 
            sample_batch_size: int = 4,
            left_margin: float = 0.08,
            right_margin: float = 0.08,
            top_p: float = 0.9,
            temperature: int = 1,
            stop_repetition: int = 3,
            kvcache: bool = True,
            smart_transcript: bool = True
            ) -> sieve.Audio:
        
        """
        Perform audio synthesis or editing using VoiceCraft.

        :param audio_path: Reference audio file (required).
        :param input_text: Text to synthesize (required). In TTS and Long TTS mode, write the text to synthesize. In edit mode, write the text to edit.
        :param mode: Mode can be TTS, Long TTS, or Edit (optional; default is "TTS").
        :param seed: Seed for reproduction (optional; default is -1).
        :param prompt_end_time: The end time of the initial reference audio to use. Optional; default is half of the input audio.
        :param edit_start_time: Edit start time in seconds (required for Edit mode). For substitution, edit_start_time should be equal to the start time of the word to be replaced.
        :param edit_end_time: Edit end time in seconds (required for Edit mode). For insertion, edit_end_time should be equal to edit_start_time. For subsitution, edit_end_time should be equal to the end time of the word to be replaced.
        :param split_text: Used for Long TTS. How sentences are split: "Newline" or "Sentence". Optional; default is Newline.
        :param selected_sentence: Select the sentence to rerun generation in Long TTS mode.
        :param previous_audio_tensors: Previous audio tensors required for rerun.
        :param sample_batch_size: Batch size for generation. The higher the number, the faster the output will be. Under the hood, the model will generate this many samples and choose the shortest one.
        :param left_margin: Margin to the left of the editing segment. Optional; default is 0.08.
        :param right_margin: Margin to the right of the editing segment. Optional; default is 0.08.
        :param top_p: Top p sampling. Optional; default is 0.9.
        :param temperature: Temperature for generation. Optional; default is 1.
        :param stop_repetition: Stop repetition parameter. if there are long silence in the generated audio, reduce the stop_repetition to 2 or 1. -1 = disabled. Optional; default is 3.
        :param kvcache: Use KV cache for faster inference. Optional; default is True.
        :param smart_transcript: If enabled, the target transcript will be automatically constructed based on the mode:
                         - In TTS and Long TTS mode, simply write the text you want to synthesize.
                         - In Edit mode, provide the text to replace the selected editing segment.
                        If disabled, you should manually write the target transcript:
                         - In TTS mode, provide the prompt transcript followed by the generation transcript.
                         - In Long TTS mode, select split by newline (SENTENCE SPLIT WON'T WORK) and begin each line with a prompt transcript.</br>
                         - In Edit mode, provide the full prompt text.</br>
        
        
        :return: Synthesized audio file.
        """
        import torchaudio  
        import os
        import torch
        from voicecraft.whisper_models import align , transcribe
        from voicecraft.utils import get_output_audio


        audio_path = reference_audio.path

        if audio_path is None:
            raise Exception("Invalid audio file!") 
        
        if mode not in ["TTS", "Long TTS", "Edit"]:
            raise ValueError("mode should be TTS, Long TTS, or Edit")
        
        if mode == "Edit" and (edit_start_time == -1.0 or edit_end_time == -1.0):
            raise Exception("edit_start_time and edit_end_time are required for Edit mode")
        
        if mode != "Edit" and (edit_end_time != -1.0 or edit_start_time != -1.0):
            raise Exception("edit_start_time and edit_end_time are only required for Edit mode! change mode to Edit")
        
        if mode == "Long TTS" and split_text not in ["Newline", "Sentence"]:
            raise ValueError("split_text should be Newline or Sentence")
        

        if prompt_end_time == -1.0:
            
            #ideal prompt_end_time for voicecraft = 7-9s, larger end times may cause oom
            info = torchaudio.info(audio_path)
            prompt_end_time = round(info.num_frames / info.sample_rate, 2) / 2.00


        transcribe_state = transcribe(audio_path,self.transcribe_model)
        transcribe_state = align(transcribe_state["transcript"], audio_path,self.TMP_PATH,self.align_model)


        if smart_transcript and (transcribe_state is None):
            raise Exception("Can't use smart transcript: whisper transcript not found")

        
        if mode == "Long TTS":
            if split_text == "Newline":
                sentences = input_text.split('\n')
            else:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(input_text.replace("\n", " "))
        
        # elif mode == "Rerun":
        #     #gets the selected sentence
        #     colon_position = selected_sentence.find(':')
        #     selected_sentence_idx = int(selected_sentence[:colon_position])
        #     sentences = [selected_sentence[colon_position + 1:]]
        else:
            sentences = [input_text.replace("\n", " ")]

        info = torchaudio.info(audio_path)
        audio_dur = info.num_frames / info.sample_rate

        audio_tensors = []

        inference_transcript = ""
        for sentence in sentences:
            decode_config = {"top_k": 0, "top_p": top_p, "temperature": temperature, "stop_repetition": stop_repetition,
                            "kvcache": int(kvcache), "codec_audio_sr": 16000, "codec_sr": 50,
                            "silence_tokens": [1388,1898,131], "sample_batch_size": sample_batch_size}
            if mode != "Edit":
                from voicecraft.inference_tts_scale import inference_one_sample

                
                #for tts and long tts mode, voicecraft requires target transcript = transcription before prompt_end + input_text. If smart_transcript disabled input_text must be = ref transcription + text to synthesise
                if smart_transcript:                
                    target_transcript = ""
                    for word in transcribe_state["words_info"]:
                        if word["end"] < prompt_end_time:
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        elif (word["start"] + word["end"]) / 2 < prompt_end_time:
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                            prompt_end_time = word["end"]
                            break
                        else:
                            break
                    target_transcript += f" {sentence}"
                else:
                    target_transcript = sentence

                inference_transcript += target_transcript + "\n"

                prompt_end_frame = int(min(audio_dur, prompt_end_time) * info.sample_rate)
                _, gen_audio = inference_one_sample(self.voicecraft_model["model"],
                                                    self.voicecraft_model["ckpt"]["config"],
                                                    self.voicecraft_model["ckpt"]["phn2num"],
                                                    self.voicecraft_model["text_tokenizer"], self.voicecraft_model["audio_tokenizer"],
                                                    audio_path, target_transcript, self.device, decode_config,
                                                    prompt_end_frame)
            else:
                from voicecraft.inference_speech_editing_scale import inference_one_sample

                #for edit mode target transcript = transcription before edit_start_time + input_text + transcription after edit_end_time. If smart_transcript disabled input_text must contain the entire transcript with changes (insetion,deletion,subsitution etc.)
                if smart_transcript:
                    target_transcript = ""
                    for word in transcribe_state["words_info"]:
                        if word["start"] < edit_start_time:
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                        else:
                            break
                    target_transcript += f" {sentence} "
                    for word in transcribe_state["words_info"]:
                        if word["end"] > edit_end_time:
                            target_transcript += word["word"] + (" " if word["word"][-1] != " " else "")
                else:
                    target_transcript = sentence

                inference_transcript += target_transcript + "\n"

                morphed_span = (max(edit_start_time - left_margin, 1 / decode_config['codec_sr']), min(edit_end_time + right_margin, audio_dur))
                mask_interval = [[round(morphed_span[0]*decode_config['codec_sr']), round(morphed_span[1]*decode_config['codec_sr'])]]
                mask_interval = torch.LongTensor(mask_interval)

                
                _, gen_audio = inference_one_sample(self.voicecraft_model["model"],
                                                    self.voicecraft_model["ckpt"]["config"],
                                                    self.voicecraft_model["ckpt"]["phn2num"],
                                                    self.voicecraft_model["text_tokenizer"], self.voicecraft_model["audio_tokenizer"],
                                                    audio_path, target_transcript, mask_interval, self.device, decode_config)
            gen_audio = gen_audio[0].cpu()
            audio_tensors.append(gen_audio)


        
        
        output_audio = get_output_audio(audio_tensors, decode_config['codec_audio_sr'])
        output_audio_path = 'temp/output.wav'
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
        with open(output_audio_path, 'wb') as f:
            f.write(output_audio)

        
        
        
        
        return sieve.Audio(path = output_audio_path)
            


