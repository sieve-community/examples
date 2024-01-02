import sieve

metadata = sieve.Metadata(
    description="The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts",
    code_url="https://github.com/sieve-community/examples/tree/main/text_generation",
    image=sieve.Image(path="mixtral_logo2.png"),
    tags=["Text", "Generation"],
    readme=open("README.md", "r").read(),
)


@sieve.Model(
    name="mixtral",
    gpu="a100",
    metadata=metadata,
    python_packages=[
        "torch",
        "scipy",
        "bitsandbytes",
        "accelerate",
        "sentencepiece",
        "vllm",
    ],
    python_version="3.11",
    cuda_version="11.8",
    run_commands=[
        "mkdir -p /root/.cache/weights",
        "wget -O /root/.cache/weights/.gitattributes https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/.gitattributes",
        "wget -O /root/.cache/weights/added_tokens.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/added_tokens.json",
        "wget -O /root/.cache/weights/README.md https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/README.md",
        "wget -O /root/.cache/weights/config.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/config.json",
        "wget -O /root/.cache/weights/generation_config.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/generation_config.json",
        "wget -O /root/.cache/weights/model-00001-of-00003.safetensors https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/model-00001-of-00003.safetensors",
        "wget -O /root/.cache/weights/model-00002-of-00003.safetensors https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/model-00002-of-00003.safetensors",
        "wget -O /root/.cache/weights/model-00003-of-00003.safetensors https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/model-00003-of-00003.safetensors",
        "wget -O /root/.cache/weights/model.safetensors.index.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/model.safetensors.index.json",
        "wget -O /root/.cache/weights/quant_config.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/quant_config.json",
        "wget -O /root/.cache/weights/special_tokens_map.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/special_tokens_map.json",
        "wget -O /root/.cache/weights/tokenizer.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/tokenizer.json",
        "wget -O /root/.cache/weights/tokenizer.model https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/tokenizer.model",
        "wget -O /root/.cache/weights/tokenizer_config.json https://huggingface.co/TheBloke/dolphin-2.6-mixtral-8x7b-AWQ/resolve/main/tokenizer_config.json",
    ],
)
class Model:
    def __setup__(self):
        from vllm import LLM

        main_directory = "/root/.cache/weights"

        self.llm = LLM(
            model=main_directory,
            quantization="AWQ",
            dtype="auto",
            gpu_memory_utilization=0.95,
            enforce_eager=True,
        )

    def __predict__(
        self,
        user_prompt: str = "What is the meaning of life?",
        system_prompt: str = "You are a helpful assistant.",
    ):
        """
        :param user_prompt: Input prompt
        :param system_prompt: System prompt
        :return: Generated text
        """

        import time
        from vllm import SamplingParams

        prompt_template = f"""<|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_prompt}<|im_end|>
        <|im_start|>assistant
        """

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16384)

        start = time.time()
        outputs = self.llm.generate(prompt_template, sampling_params=sampling_params)

        print(f"Time taken: {time.time() - start:.2f}s")

        generated_text = outputs[0].outputs[0].text
        generated_tokens = outputs[0].outputs[0].token_ids

        print("tok/s:", len(generated_tokens) / (time.time() - start))

        return generated_text
