import sieve

metadata = sieve.Metadata(
    title="Video Co-Tracker",
    description="Track any point in a video",
    code_url="https://github.com/sieve-community/examples/tree/main/co-tracker",
    image=sieve.Image(
        url="https://github.com/facebookresearch/co-tracker/raw/main/assets/bmx-bumps.gif"
    ),
    tags=["Video", "Tracking"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="co-tracker", 
    gpu=True, 
    python_packages=[
        "git+https://github.com/facebookresearch/co-tracker",
        "torchvision==0.15.2",
        "torch==2.0.1",
        "einops==0.4.1",
        "timm==0.6.7",
        "tqdm==4.64.1",
        "flow_vis",
        "matplotlib==3.7.0",
        "moviepy==1.0.3",
    ],
    python_version="3.11",
    cuda_version="11.8",
    run_commands=[
        "mkdir -p /root/.cache/torch/hub/facebookresearch_co-tracker_master",
        "wget -q https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth -O /root/.cache/torch/hub/facebookresearch_co-tracker_master/cotracker_stride_4_wind_8.pth",
    ],
    metadata=metadata
)
class CoTracker:
    def __setup__(self):
        import torch
        import os
        import tempfile
        from cotracker.predictor import CoTrackerPredictor
        self.model = CoTrackerPredictor(
            checkpoint='/root/.cache/torch/hub/facebookresearch_co-tracker_master/cotracker_stride_4_wind_8.pth'
            )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
        self.videos_dir = tempfile.mkdtemp()
    
    def __predict__(self, video: sieve.Video, grid_size: int = 30, pad_size: int = 100, visualize: bool = False):
        from cotracker.utils.visualizer import Visualizer, read_video_from_path
        import torch
        import os
        
        loaded_video = read_video_from_path(video.path)
        loaded_video = torch.from_numpy(loaded_video).permute(0, 3, 1, 2).float()

        if self.use_cuda:
            loaded_video = loaded_video.cuda()

        # Unsqueeze to add batch dimension
        loaded_video = loaded_video.unsqueeze(0)
        
        pred_tracks, pred_visibility = self.model(loaded_video, grid_size=grid_size)

        for file in os.listdir(self.videos_dir):
            os.remove(os.path.join(self.videos_dir, file))

        sieve_array_pred_tracks = sieve.Array(array=pred_tracks.cpu().numpy())
        sieve_array_pred_visibility = sieve.Array(array=pred_visibility.cpu().numpy())

        if visualize:

            vis = Visualizer(
                save_dir=self.videos_dir,
                pad_value=pad_size
            )

            vis.visualize(
                video=loaded_video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename="output",
            )

            return sieve.Video(path=os.path.join(self.videos_dir, "output_pred_track.mp4"))

        return sieve_array_pred_tracks, sieve_array_pred_visibility
            
