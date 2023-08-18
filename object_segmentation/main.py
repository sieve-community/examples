import sieve

metadata = sieve.Metadata(
    title="Segment Objects in Video",
    description="Run hyperparallelized object segmentation on video.",
    code_url="https://github.com/sieve-community/examples/blob/main/object_segmentation/main.py",
    tags=["Segmentation", "Video"],
    readme=open("README.md", "r").read(),
)


@sieve.workflow(name="object-segmentation", metadata=metadata)
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
    sieve.push(
        workflow="object-segmentation",
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/bike.mp4"
            }
        },
    )
