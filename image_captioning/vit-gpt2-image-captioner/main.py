import sieve
from typing import Dict, List

metadata = sieve.Metadata(
    description="Generate captions for an image with ViT and GPT-2.",
    code_url="https://github.com/sieve-community/examples/tree/main/image_understanding/vit-gpt2-image-captioner",
    image=sieve.Image(
        url="https://ankur3107.github.io/assets/images/vision-encoder-decoder.png"
    ),
    tags=["Captioning", "Image"],
    readme=open("README.md", "r").read(),
)


@sieve.Model(
    name="vit-gpt2-image-captioner",
    gpu=True,
    python_version="3.8",
    python_packages=["torch==1.8.1", "transformers==4.23.1"],
    metadata=metadata,
)
class ImageCaptioner:
    def __setup__(self):
        import torch
        from transformers import (
            AutoTokenizer,
            ViTFeatureExtractor,
            VisionEncoderDecoderModel,
        )

        device = "cuda"
        encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(
            device
        )
        self.clean_text = lambda x: x.replace("<|endoftext|>", "").split("\n")[0]

    def __predict__(self, image: sieve.Image) -> str:
        """
        :param image: Image to caption
        :return: Generated caption of the image
        """
        image = self.feature_extractor(image.array, return_tensors="pt").pixel_values.to(
            "cuda"
        )
        caption_ids = self.model.generate(image, max_length=64)[0]
        caption_text = self.clean_text(self.tokenizer.decode(caption_ids))
        return caption_text


wf_metadata = sieve.Metadata(
    title="Caption an Image",
    description="Understand and generate captions for images.",
    code_url="https://github.com/sieve-community/examples/tree/main/image_captioning/main.py",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/image_captioning/cover.jpg"
    ),
    tags=["Generative", "Image"],
    readme=open("README.md", "r").read(),
)
