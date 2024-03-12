import sieve

metadata = sieve.Metadata(
    description="InternLM-XComposer2 is a groundbreaking vision-language large model (VLLM) based on InternLM2-7B excelling in free-form text-image composition and comprehension",
    code_url="https://github.com/sieve-community/examples/blob/main/visual_qa/internlmx",
    tags=["Visual", "Image", "VQA", "Dialogue"],
    image=sieve.Image(path="logo_en_crop.png"),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    gpu=sieve.gpu.A100(split=3),
    name="internlmx-composer-2q",
    python_packages=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers==4.33.2",
        "sentencepiece==0.1.99",
        "gradio==4.13.0",
        "markdown2==2.4.10",
        "xlsxwriter==3.1.2",
        "einops",
        "auto_gptq",
    ],
    python_version="3.11",
    cuda_version="11.8",
    metadata=metadata,
)
class InternLMX:
    def __setup__(self):
        import torch, auto_gptq
        from transformers import AutoModel, AutoTokenizer
        from auto_gptq.modeling import BaseGPTQForCausalLM

        auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
        torch.set_grad_enabled(False)

        class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
            layers_block_name = "model.layers"
            outside_layer_modules = [
                "vit",
                "vision_proj",
                "model.tok_embeddings",
                "model.norm",
                "output",
            ]
            inside_layer_modules = [
                ["attention.wqkv.linear"],
                ["attention.wo.linear"],
                ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                ["feed_forward.w2.linear"],
            ]

        self.model = InternLMXComposer2QForCausalLM.from_quantized(
            "internlm/internlm-xcomposer2-vl-7b-4bit",
            trust_remote_code=True,
            device="cuda:0",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "internlm/internlm-xcomposer2-vl-7b-4bit", trust_remote_code=True
        )

    def __predict__(
        self, image: sieve.File, prompt: str = "Describe this image in detail"
    ):
        """
        :param image: Image to process
        :param prompt: Prompt with which to process the image
        :return: Response from the model
        """
        import torch
        import time

        start_time = time.time()
        query = f"<ImageHere>{prompt}"
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(
                self.tokenizer,
                query=query,
                image=image.path,
                history=[],
                do_sample=False,
            )
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return response
