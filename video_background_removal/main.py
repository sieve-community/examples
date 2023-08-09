import sieve
from bg import U2NetMask, U2NetBlur


@sieve.workflow(name="video_background_mask")
def background_mask(video: sieve.Video) -> sieve.Video:
    images = sieve.reference("sieve/video-splitter")(video)
    masks = U2NetMask()(images)
    return sieve.reference("sieve/frame-combiner")(masks)


@sieve.workflow(name="video_background_blur")
def background_blur(video: sieve.Video) -> sieve.Video:
    images = sieve.reference("sieve/video-splitter")(video)
    masks = U2NetBlur()(images)
    return sieve.reference("sieve/frame-combiner")(masks)


if __name__ == "__main__":
    sieve.push(
        background_mask,
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/bike.mp4"
            }
        },
    )
