import sieve
from typing import Dict
import uuid


@sieve.Model(
    name="wav2lip",
    gpu=True,
    python_packages=[
        "numpy==1.23.5",
        "requests==2.28.1",
        "imageio[ffmpeg]==2.22.4",
        "librosa==0.9.2",
        "numba==0.56.4",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "mediapipe==0.9.0.1",
    ],
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libavcodec58",
        "ffmpeg",
        "ninja-build",
    ],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models/wav2lip",
        "wget -c https://storage.googleapis.com/sieve-public-model-assets/wav2lip/wav2lip.pth -O /root/.cache/models/wav2lip/wav2lip.pth",
    ],
    iterator_input=True,
)
class Wav2Lip:
    def __setup__(self):
        from utils import Model

        self.model = Model("/root/.cache/models/wav2lip/wav2lip.pth")

    def __predict__(self, video: sieve.Video, audio: sieve.Audio, faces_dict: Dict):
        for vid, aud, faces in zip(video, audio, faces_dict):
            longest_face_id = None
            for face_id, face in faces.items():
                if longest_face_id is None or len(face) > len(faces[longest_face_id]):
                    longest_face_id = face_id

            faces = faces[longest_face_id]
            face_boxes = [x["box"] for x in faces]
            interpolated_faces = [
                (int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in face_boxes
            ]

            audio_file = aud.path
            video_file = vid.path
            output_filename = f"result_voice-{uuid.uuid4()}.mp4"
            output_filename = self.model.predict(
                video_file,
                audio_file,
                output_filename,
                1,
                interpolated_faces,
                faces[0]["frame_number"],
            )
            yield sieve.Video(path=output_filename)
