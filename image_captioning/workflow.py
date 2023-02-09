import sieve
from typing import List
from main import ImageCaptioner


@sieve.workflow(name="image_captioning")
def image_captioning(image: sieve.Image) -> List:
    captioner = ImageCaptioner()
    return captioner(image)
