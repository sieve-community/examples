import sieve


@sieve.function(name="pyscenedetect", python_packages=["scenedetect[opencv]"])
def scene_detection(video: sieve.Video) -> list:
    """
    :param video: The video to detect scenes in
    :return: A list of start and end times for each scene
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
            {
                "start_seconds": start_time_in_seconds,
                "end_seconds": end_time_in_seconds,
                "scene_number": i + 1,
            }
        )

    video_manager.release()

    return scenes


@sieve.workflow(name="scene_change_detection")
def wf(video: sieve.Video) -> list:
    """
    :param video: The video to detect scenes in
    :return: A list of start and end times for each scene
    """

    return scene_detection(video)


if __name__ == "__main__":
    sieve.push(
        workflow="scene_change_detection",
        inputs={
            "video": {
                "url": "https://storage.googleapis.com/sieve-public-videos-grapefruit/boston-robot.mp4"
            }
        },
    )
