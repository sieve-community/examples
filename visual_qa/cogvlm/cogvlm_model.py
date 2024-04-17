import torch
import sieve

metadata = sieve.Metadata(
    description="Powerful open-source visual language model (VLM) supporting image understanding and multi-turn dialogue",
    code_url="https://github.com/sieve-community/examples/blob/main/visual_qa/cogvlm",
    tags=["Visual", "Image", "VQA", "Dialogue"],
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/cogvlm-icon.jpg"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="cogvlm-chat",
    gpu=sieve.gpu.L4(),
    python_packages=[
        "SwissArmyTransformer>=0.4.8",
        "transformers>=4.33.1",
        "xformers>=0.0.22",
        "torch>1.10.0",
        "torchvision",
        "spacy>=3.6.0",
        "scipy",
        "pillow>=10.0.1",
        "deepspeed>=0.11.0",
        "seaborn",
        "loguru~=0.7.2",
        "accelerate",
        "bitsandbytes",
    ],
    python_version="3.11",
    cuda_version="11.8",
    metadata=metadata,
    run_commands=[
        "mkdir -p /root/.cache/weights",
        "mkdir -p /root/.cache/vicuna_weights",
        "wget -O /root/.cache/vicuna_weights/.gitattributes https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/.gitattributes",
        "wget -O /root/.cache/vicuna_weights/README.md https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/README.md",
        "wget -O /root/.cache/vicuna_weights/config.json https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/config.json",
        "wget -O /root/.cache/vicuna_weights/generation_config.json https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/generation_config.json",
        "wget -O /root/.cache/vicuna_weights/pytorch_model-00001-of-00002.bin https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00001-of-00002.bin",
        "wget -O /root/.cache/vicuna_weights/pytorch_model-00002-of-00002.bin https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model-00002-of-00002.bin",
        "wget -O /root/.cache/vicuna_weights/pytorch_model.bin.index.json https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/pytorch_model.bin.index.json",
        "wget -O /root/.cache/vicuna_weights/special_tokens_map.json https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/special_tokens_map.json",
        "wget -O /root/.cache/vicuna_weights/tokenizer.model https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer.model",
        "wget -O /root/.cache/vicuna_weights/tokenizer_config.json https://huggingface.co/lmsys/vicuna-7b-v1.5/resolve/main/tokenizer_config.json",
        "wget -O /root/.cache/weights/.gitattributes https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/.gitattributes",
        "wget -O /root/.cache/weights/README.md https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/README.md",
        "wget -O /root/.cache/weights/config.json https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/config.json",
        "wget -O /root/.cache/weights/configuration_cogvlm.py https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/configuration_cogvlm.py",
        "wget -O /root/.cache/weights/generation_config.json https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/generation_config.json",
        "wget -O /root/.cache/weights/model-00001-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00001-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00002-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00002-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00003-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00003-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00004-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00004-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00005-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00005-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00006-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00006-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00007-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00007-of-00008.safetensors",
        "wget -O /root/.cache/weights/model-00008-of-00008.safetensors https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model-00008-of-00008.safetensors",
        "wget -O /root/.cache/weights/model.safetensors.index.json https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/model.safetensors.index.json",
        "wget -O /root/.cache/weights/modeling_cogvlm.py https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/modeling_cogvlm.py",
        "wget -O /root/.cache/weights/visual.py https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/visual.py",
    ],
)
class Model:
    def __setup__(self):
        from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoConfig
        from pathlib import Path
        import os

        print(os.listdir("/root/.cache/weights"))
        config = AutoConfig.from_pretrained(
            "/root/.cache/weights/", trust_remote_code=True
        )
        model_directory = Path("/root/.cache/weights")
        tokenizer_directory = Path("/root/.cache/vicuna_weights")
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_directory, config="/root/.cache/vicuna_weights"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
            config=config,
        ).eval()

        # Move the model to the CUDA device
        # self.model.to("cuda")

    def __predict__(
        self,
        image: sieve.Image,
        prompt: str = "Caption this image",
        vqa_mode: bool = False,
    ):
        """
        :param image: Input image
        :param prompt: Input prompt
        :param vqa_mode: Whether to use the VQA template (more concise output)
        """

        from PIL import Image
        import torch

        image = Image.open(image.path).convert("RGB")
        if vqa_mode:
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=prompt,
                history=[],
                images=[image],
                template_version="vqa",
            )
        else:
            inputs = self.model.build_conversation_input_ids(
                self.tokenizer, query=prompt, history=[], images=[image]
            )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[inputs["images"][0].to("cuda").to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            print(self.tokenizer.decode(outputs[0])[:-4])
            return self.tokenizer.decode(outputs[0])[:-4]
