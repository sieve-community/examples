import sieve
from pydantic import BaseModel
from typing import List

model_metadata = sieve.Metadata(
    description="Detect scene changes in a video with PySceneDetect.",
    code_url="https://github.com/sieve-community/examples/blob/main/video_scene_detection/main.py",
    image=sieve.Image(
        url="https://raw.githubusercontent.com/Breakthrough/PySceneDetect/main/website/pages/img/pyscenedetect_logo_small.png"
    ),
    tags=["Detection", "Video"],
    readme=open("README.md", "r").read(),
)

class Scene(BaseModel):
    start_seconds: float
    end_seconds: float
    start_timecode: str
    end_timecode: str
    scene_number: int

@sieve.function(
    name="pyscenedetect",
    python_packages=[
        "scenedetect[opencv]",
        "opencv-python-headless==4.5.5.64",
    ],
    system_packages=["libgl1"],
    metadata=model_metadata,
)
def scene_detection(video: sieve.Video) -> List[Scene]:
    """
    :param video: The video to detect scenes in
    :return: A list of scenes
    """

    from scenedetect.detectors import ContentDetector
    from scenedetect.scene_manager import SceneManager
    from scenedetect.video_manager import VideoManager

    video_manager = VideoManager([video.path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)
    print("List of scenes obtained:")
    scenes = []
    for i, scene in enumerate(scene_list):
        print(
            "Scene %2d: Start %s / Frame %d, End %s / Frame %d"
            % (
                i + 1,
                scene[0].get_timecode(),
                scene[0].get_frames(),
                scene[1].get_timecode(),
                scene[1].get_frames(),
            )
        )

        start_time_in_seconds = scene[0].get_frames() / video.fps
        end_time_in_seconds = scene[1].get_frames() / video.fps
        scenes.append(
            Scene(
                start_seconds=start_time_in_seconds,
                end_seconds=end_time_in_seconds,
                scene_number=i + 1,
                start_timecode=scene[0].get_timecode(),
                end_timecode=scene[1].get_timecode(),
            )
        )

    video_manager.release()

    return scenes

if __name__ == "__main__":
    print(scene_detection.run(
        sieve.Video(path="/Users/Mokshith/Desktop/gordon.mp4")
    ))