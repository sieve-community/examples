import sieve
from main import ImageCaptioner

@sieve.workflow(name="image-captioning")
def image_captioning(image: sieve.Image) -> str:
    '''
    :param img: Image to caption
    :return: Generated caption of the image
    '''
    captioner = ImageCaptioner()
    return captioner(image)
