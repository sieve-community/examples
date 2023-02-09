import sieve
from combine import frame_combine
from splitter import VideoSplitter
from segmentation import InstanceSegmentation


class ImageWithFrameNumber(sieve.Image):
    frame_number: int


@sieve.workflow(name="object-segmentation")
def segmentation(a: sieve.Video):
    m = VideoSplitter(a)
    return frame_combine(InstanceSegmentation()(m))


if __name__ == "__main__":
    sieve.push(
        segmentation,
        {
            "a": sieve.Video(
                url="https://storage.googleapis.com/sieve-test-videos-central/01-lebron-dwade.mp4"
            )
        },
    )
