import sieve
from wav2lip import Wav2Lip


# Wav2Lip takes in a video, audio, and a a set of faces that are tracked over time.
@sieve.workflow(name="video_lipsyncing")
def wav2lip(video: sieve.Video, audio: sieve.Audio):
    images = sieve.reference("sieve-developer/video-splitter")(video)
    faces = sieve.reference("sieve-developer/mediapipe-face-detector")(images)
    tracked_faces = sieve.reference("sieve-developer/sort-tracker")(faces)
    return Wav2Lip()(video, audio, tracked_faces)
