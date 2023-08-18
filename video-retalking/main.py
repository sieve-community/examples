import os
import sieve
from pydantic import BaseModel
 
class LipsyncInputs(BaseModel):
    video: sieve.Video
    audio: sieve.Audio


base_path = "/root/.cache/retalking"
class Args:
    def __init__(self,
            DNet_path=os.path.join(base_path, 'DNet.pt'),
            LNet_path=os.path.join(base_path, 'LNet.pth'),
            ENet_path=os.path.join(base_path, 'ENet.pth'),
            face3d_net_path='/root/.cache/retalking/face3d_pretrain_epoch_20.pth',
            face=None,
            audio=None,
            exp_img='neutral',
            outfile=None,
            fps=25.,
            pads=[0, 20, 0, 0],
            face_det_batch_size=4,
            LNet_batch_size=16,
            img_size=384,
            crop=[0, -1, 0, -1],
            box=[-1, -1, -1, -1],
            nosmooth=False,
            static=False,
            up_face='original',
            one_shot=False,
            without_rl1=False,
            tmp_dir='temp',
            re_preprocess=False):
        self.DNet_path = DNet_path
        self.LNet_path = LNet_path
        self.ENet_path = ENet_path
        self.face3d_net_path = face3d_net_path
        self.face = face
        self.audio = audio
        self.exp_img = exp_img
        self.outfile = outfile
        self.fps = fps
        self.pads = pads
        self.face_det_batch_size = face_det_batch_size
        self.LNet_batch_size = LNet_batch_size
        self.img_size = img_size
        self.crop = crop
        self.box = box
        self.nosmooth = nosmooth
        self.static = static
        self.up_face = up_face
        self.one_shot = one_shot
        self.without_rl1 = without_rl1
        self.tmp_dir = tmp_dir
        self.re_preprocess = re_preprocess

import os
import sieve

@sieve.Model(
    name="video_retalker",
    gpu=True,
    python_packages=[
        "torch==1.13.1", "cmake==3.26.3", "face-alignment==1.3.5",
        "ninja==1.10.2.3", "einops==0.4.1", "facexlib==0.2.5",
        "librosa==0.9.2", "gradio>=3.7.0", "numpy==1.23.1",
        "mediapipe==0.9.0.1"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    machine_type="a100",
    run_commands=[
        "pip install basicsr",
        "pip install dlib",
        "pip install kornia",
        "mkdir -p /root/.cache/retalking",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth -O /root/.cache/retalking/30_net_gen.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt -O /root/.cache/retalking/DNet.pt",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth -O /root/.cache/retalking/ENet.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat -O /root/.cache/retalking/expression.mat",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth -O /root/.cache/retalking/face3d_pretrain_epoch_20.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth -O /root/.cache/retalking/GFPGANv1.3.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth -O /root/.cache/retalking/GPEN-BFR-512.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth -O /root/.cache/retalking/LNet.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth -O /root/.cache/retalking/ParseNet-latest.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth -O /root/.cache/retalking/RetinaFace-R50.pth",
        "wget -c https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O /root/.cache/retalking/shape_predictor_68_face_landmarks.dat",
        "mkdir -p /root/.cache/retalking/BFM",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/01_MorphableModel.mat -O /root/.cache/retalking/BFM/01_MorphableModel.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/BFM_exp_idx.mat -O /root/.cache/retalking/BFM/BFM_exp_idx.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/BFM_front_idx.mat -O /root/.cache/retalking/BFM/BFM_front_idx.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/BFM_model_front.mat -O /root/.cache/retalking/BFM/BFM_model_front.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/Exp_Pca.bin -O /root/.cache/retalking/BFM/Exp_Pca.bin",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/facemodel_info.mat -O /root/.cache/retalking/BFM/facemodel_info.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/select_vertex_id.mat -O /root/.cache/retalking/BFM/select_vertex_id.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/similarity_Lm3D_all.mat -O /root/.cache/retalking/BFM/similarity_Lm3D_all.mat",
        "wget -c https://storage.googleapis.com/mango-public-models/retalking/BFM/std_exp.txt -O /root/.cache/retalking/BFM/std_exp.txt",
    ],
    environment_variables=[
        sieve.Env(
            name="stabilize_expression",
            description="whether or not to stabilize the expression before lip-syncing",
            default="false"
        ),
        sieve.Env(
            name="reference_enhance",
            description="whether or not to enhance the reference image before lip-syncing",
            default="false"
        ),
        sieve.Env(
            name="gfpgan_enhance",
            description="whether or not to enhance the generated lipsync segment",
            default="true"
        ),
        sieve.Env(
            name="post_enhance",
            description="whether or not to enhance the image as a whole after the lipsyncing has been added back",
            default="true"
        )
    ]
)
class VideoRetalker():
    def __setup__(self):
        from inference import setup
        args = Args(
            face="something.mp4",
            audio="something.mp3",
            outfile="./output.mp4"
        )
        self.enhancer, self.restorer, self.kp_extractor, self.D_Net, self.model, self.croper, self.lm3d_std, self.expression, self.device = setup(args, base_dir=base_path)

    def __predict__(self, inputs: LipsyncInputs):
        import time
        start_inference = time.time()
        from inference import predict
        face_video = inputs.video
        audio_file = inputs.audio
        args = Args(
            face=face_video.path,
            audio=audio_file.path,
            outfile="./output.mp4"
        )

        stabilize_expression = os.environ.get("stabilize_expression") == "true"
        reference_enhance = os.environ.get("reference_enhance") == "true"
        gfpgan_enhance = os.environ.get("gfpgan_enhance") == "true"
        post_enhance = os.environ.get("post_enhance") == "true"
        print("stabilize_expression: ", stabilize_expression)
        print("reference_enhance: ", reference_enhance)
        print("gfpgan_enhance: ", gfpgan_enhance)
        print("post_enhance: ", post_enhance)
        out_file = predict(
            args,
            self.enhancer,
            self.restorer,
            self.kp_extractor,
            self.D_Net,
            self.model,
            self.croper,
            self.lm3d_std,
            self.expression,
            self.device,
            stabilize_expression=stabilize_expression,
            reference_enhance=reference_enhance,
            gfp_enhance=gfpgan_enhance,
            post_enhance=post_enhance,
        )
        end_inference = time.time()
        print("Total inference time: ", end_inference - start_inference)
        return sieve.Video(path=out_file)

@sieve.workflow(name="video_retalking")
def video_retalking(inputs: LipsyncInputs) -> sieve.Video:
    retalker = VideoRetalker()
    return retalker(inputs)