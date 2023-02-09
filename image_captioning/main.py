import sieve
from typing import Dict, List
import torch
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 

@sieve.Model(
    name="vit-gpt2-image-captioner",
    gpu = True,
    python_version="3.8",
    python_packages=[
        'torch==1.8.1',
        'transformers==4.23.1'
    ]
)
class ImageCaptioner:
    def __setup__(self):
        device='cuda'
        encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)
        self.clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]

    def __predict__(self, img: sieve.Image) -> str:
        image = self.feature_extractor(img.array, return_tensors="pt").pixel_values.to('cuda')
        caption_ids = self.model.generate(image, max_length = 64)[0]
        caption_text = self.clean_text(self.tokenizer.decode(caption_ids))
        return caption_text
