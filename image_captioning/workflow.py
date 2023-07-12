import sieve
from main import ImageCaptioner


@sieve.workflow(name="image-captioning")
def image_captioning(image: sieve.Image) -> str:
    captioner = ImageCaptioner()
    return captioner(image)
