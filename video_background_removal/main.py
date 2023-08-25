import sieve
from bg import U2Net

metadata = sieve.Metadata(
    title="Remove Background from Video",
    description="Remove the background from a video with U2Net.",
    code_url="https://github.com/sieve-community/examples/tree/main/video_background_removal/main.py",
    image=sieve.Image(
        url="https://storage.googleapis.com/sieve-public-data/video_background_mask/cover.gif"
    ),
    tags=["Video", "Masking"],
    readme=open("README.md", "r").read(),
)


@sieve.workflow(name="video_background_removal", metadata=metadata)
def background_mask(video: sieve.Video, blur: bool) -> sieve.Video:
    images = sieve.reference("sieve/video-splitter")(video)
    masks = U2Net()(images, blur)
    return sieve.reference("sieve/frame-combiner")(masks)


if __name__ == "__main__":
    sieve.push(
        background_mask,
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/bike.mp4"
            },
            "blur": True,
        },
    )
