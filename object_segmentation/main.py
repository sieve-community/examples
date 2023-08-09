import sieve


@sieve.workflow(name="object-segmentation")
def segmentation(video: sieve.Video) -> sieve.Video:
    from segmentation import InstanceSegmentation

    """
    :param video: Video to segment objects from
    :return: Original video with segmented masks overlayed
    """
    frames = sieve.reference("sieve/video-splitter")(video)
    segmented_frames = InstanceSegmentation()(frames)
    return sieve.reference("sieve/frame-combiner")(segmented_frames)


if __name__ == "__main__":
    sieve.push(workflow="object-segmentation", inputs={"video": {"url": ""}})
