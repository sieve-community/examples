import re
import torch
from PIL import Image
import sieve

metadata = sieve.Metadata(
    description="VILA is a visual language model (VLM) pretrained with interleaved image-text data at scale, enabling video understanding and multi-image understanding capabilities.",
    code_url="https://github.com/sieve-community/examples/blob/main/visual_qa/vila",
    tags=["Visual", "Image", "VQA", "Dialogue"],
    image=sieve.Image(
        path="vila_logo.jpeg"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="vila",
    gpu=sieve.gpu.A100(split=2),
    python_version="3.10",
    cuda_version="11.8",
    python_packages=["accelerate"],
    system_packages=["git-lfs"],
    run_commands=[
        "wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -q -O /root/.cache/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        "pip install /root/.cache/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        "pip install git+https://github.com/Efficient-Large-Model/VILA.git",
        "pip install protobuf",
        "pip install git+https://github.com/huggingface/transformers@v4.36.2",
        "site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')",
        "mkdir -p /root/.cache/repo/vila",
        "mkdir -p /root/.cache/models/vila1.5-3b",
        "git clone https://github.com/Efficient-Large-Model/VILA.git /root/.cache/repo/vila",
        "cp -rv /root/.cache/repo/vila/llava/train/transformers_replace/* $site_pkg_path/transformers/",
        "git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-3b /root/.cache/models/vila1.5-3b",
    ],
    metadata=metadata
)
class VILA:
    def __setup__(self):
        from llava.mm_utils import (get_model_name_from_path)
        from llava.model.builder import load_pretrained_model

        model_path = "/root/.cache/models/vila1.5-3b"

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, self.model_name, None)

    def __predict__(self, input_file: sieve.File, query: str = "Describe this", sampling_strategy: str = "auto", sampling_count: int = 6):    
        """
        :param input_file: Input file
        :param query: Query to be passed to the model
        :param sampling_strategy: Sampling strategy for video frames. Default is "auto", which samples frames uniformly across the video. Can also be comma-separated frame indices.
        :param sampling_count: Number of frames to sample from the video. Default is 6. Only used if sampling_strategy is "auto" and the input file is a video.

        """
        from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                    DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                                    IMAGE_TOKEN_INDEX)
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import (KeywordsStoppingCriteria,
                                    process_images, tokenizer_image_token)
        from llava.utils import disable_torch_init  
        from custom_utils import get_frame_from_vcap
        import time
        from decord import VideoReader

        video_file = None
        image_file = None

        # check if input_file is a video or image by checking the extension
        if input_file.path.endswith(('.mp4', '.avi', '.mov')):
            video_file = input_file.path
        else:
            image_file = input_file.path
        
        # if input does not match any of the above, raise an error
        if video_file is None and image_file is None:
            raise ValueError("Input file must be either a video or an image")

        # inference params
        sep, temperature, top_p, num_beams, max_new_tokens, conv_mode = ",", 0.2, None, 1, 512, "vicuna_v1"

        disable_torch_init()
        if video_file is None:
            images = Image.open(image_file).convert("RGB")
        else:
            vidcap = VideoReader(video_file)
            images =  get_frame_from_vcap(vidcap, sampling_count, sampling_strategy=sampling_strategy)

        qs = query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        inferred_conv_mode = "llava_v0"  # Set a default conversation mode
        if "llama-2" in self.model_name.lower():
            inferred_conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            inferred_conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            inferred_conv_mode = "mpt"

        if conv_mode is not None and inferred_conv_mode != conv_mode:
            pass
        else:
            conv_mode = inferred_conv_mode

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        start_time = time.time()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        output_length = output_ids.size(1)
        print("tok/sec: ", output_length / (time.time() - start_time))
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs