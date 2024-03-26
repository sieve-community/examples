import sieve

model_metadata = sieve.Metadata(
    description="Translate text into 96 different languages",
    code_url="https://github.com/sieve-community/examples/blob/main/translation/seamless_text_to_text",
    image=sieve.Image(
        url="https://github.com/facebookresearch/seamless_communication/raw/main/seamlessM4T.png"
    ),
    tags=["Translation", "Text"],
    readme=open("README.md", "r").read(),
)


### Text 2 Text
@sieve.Model(
    name="seamless_text2text",
    gpu=sieve.gpu.L4(),
    python_packages=["git+https://github.com/facebookresearch/seamless_communication@711707abb077efcec664888290904700c8f7b680"],
    system_packages=[
        "libsndfile1",
        "libopenblas-base",
        "libgomp1",
        "ffmpeg",
    ],
    run_commands=[
        # "mkdir -p /root/.cache/torch/hub/fairseq2/assets/checkpoints/43b8b74ddb6b78486fb47754",
        # "wget -q https://huggingface.co/facebook/seamless-m4t-large/resolve/main/multitask_unity_large.pt -O /root/.cache/torch/hub/fairseq2/assets/checkpoints/43b8b74ddb6b78486fb47754/multitask_unity_large.pt",
    ],
    cuda_version="11.8",
    metadata=model_metadata,
)
class SeamlessText2Text:
    def __setup__(self):
        import torch
        from seamless_communication.inference.translator import Translator
        import torchaudio

        # Initialize a Translator object with a multitask model, vocoder on the GPU.
        self.translator = Translator(
            "seamlessM4T_v2_large", "vocoder_36langs", device=torch.device("cuda"), dtype=torch.float16
        )

    def __predict__(self, text: str, source_language: str, target_language: str) -> str:
        """
        :param text: Text to translate
        :param source_language: Source language. Try "eng" for English, "spa" for Spanish, or others below.
        :param target_language: Target language. Try "eng" for English, "spa" for Spanish, or others below.
        :return: Translated text
        """

        # Translate the text to desired language in text
        translated_text, _, = self.translator.predict(
            text, "t2tt", target_language, src_lang=source_language
        )

        # Convert text to string
        translated_text = ' '.join([str(elem) for elem in translated_text])

        return translated_text
