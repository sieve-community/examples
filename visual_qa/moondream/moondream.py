import sieve
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-04-02"

metadata = sieve.Metadata(
    description="a tiny vision language model that kicks ass and runs anywhere",
    code_url="https://github.com/sieve-community/examples/blob/main/visual_qa/moondream",
    tags=["Visual", "Image", "VQA", "Dialogue"],
    image=sieve.Image(
        url="https://raw.githubusercontent.com/vikhyat/moondream/github-pages/favicon.png"
    ),
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    gpu=sieve.gpu.L4(split=3),
    name="moondream",
    python_version="3.11",
    python_packages=[
        "torch>=2.1.1",
        "transformers",
        "timm",
        "einops",
        "Pillow"
    ],
    metadata=metadata
)
class MoonDream:
    def __setup__(self):
        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=REVISION,
        ).to(device="cuda", dtype=torch.float16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

    def __predict__(
        self,
        image: sieve.File,
        question: str = "Describe this image."
    ):
        """
        :param image: Image to process
        :param question: Question to ask about the image
        :return: The output of the model.
        """
        image = Image.open(image.path)
        enc_image = self.model.encode_image(image)
        return self.model.answer_question(enc_image, question, self.tokenizer)
