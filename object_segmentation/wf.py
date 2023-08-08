import sieve
from segmentation import InstanceSegmentation


@sieve.workflow(name="object-segmentation")
def segmentation(video: sieve.Video) -> sieve.Video:
    """
    :param video: Video to segment objects from
    :return: Original video with segmented masks overlayed
    """
    frames = sieve.reference("sieve/video-splitter")(video)
    segmented_frames = InstanceSegmentation()(frames)
    return sieve.reference("sieve/frame-combiner")(segmented_frames)
