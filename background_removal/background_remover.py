import sieve

metadata = sieve.Metadata(
    description="Remove background from image and video",
    code_url="https://github.com/sieve-community/examples/tree/main/background_removal",
    image=sieve.Image(
        url="https://github.com/xuebinqin/DIS/raw/main/figures/dis5k-v1-sailship.jpeg"
    ),
    tags=["Video", "Background", "Removal"],
    readme=open("README.md", "r").read(),
)

@sieve.Model(
    name="dis_background_remover",
    gpu = "a100-20gb",
    python_packages=[
        "six==1.16.0",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "opencv-python-headless==4.5.5.64"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/dis/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/isnet-general-use.pth' -P /root/.cache/dis/models/"
    ],
    metadata=metadata
)
class VideoBackgroundRemover:
    def __setup__(self):
        import numpy as np
        import cv2
        from skimage import io

        import torch
        from torchvision.transforms.functional import normalize
        import torch.nn.functional as F
        from isnet import ISNetDIS
        model_path = '/root/.cache/dis/models/isnet-general-use.pth'
        self.input_size=[1024,1024]
        self.net = ISNetDIS()

        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_path))
            self.net = self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(model_path,map_location="cpu"))
        self.net.eval()
    
    def __process_frame__(self, image_array, return_mask=False):
        import numpy as np
        import cv2
        from skimage import io

        import torch
        from torchvision.transforms.functional import normalize
        import torch.nn.functional as F
        import uuid
        import time

        t1 = time.time()
        # im = io.imread(input_image.path)
        im = image_array
        # print("Time taken to read image: ", time.time() - t1)
        t1 = time.time()
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        # print("Time taken to check image shape: ", time.time() - t1)
        t1 = time.time()
        im_tensor = torch.from_numpy(im.transpose((2, 0, 1))).float()
        # print("Time taken to convert image to tensor: ", time.time() - t1)
        t1 = time.time()
        im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), self.input_size, mode="bilinear").type(torch.uint8)
        # print("Time taken to upsample image: ", time.time() - t1)
        t1 = time.time()
        image = torch.divide(im_tensor,255.0)
        # print("Time taken to divide image: ", time.time() - t1)
        t1 = time.time()
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
        # print("Time taken to normalize image: ", time.time() - t1)
        t1 = time.time()

        if torch.cuda.is_available():
            image=image.cuda()
        result = self.net(image)
        # print("Time taken to predict: ", time.time() - t1)
        t1 = time.time()
        result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
        # print("Time taken to squeeze and upsample: ", time.time() - t1)
        t1 = time.time()
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        # print("Time taken to normalize result: ", time.time() - t1)
        t1 = time.time()
        im_mask = (result*255).permute(1,2,0).cpu().data.numpy()
        threshold = 127.5
        im_mask = np.where(im_mask > threshold, 255, 0).astype(np.uint8)
        # print("Time taken to convert mask to numpy: ", time.time() - t1)
        t1 = time.time()
        # print("Time taken to post process: ", time.time() - t1)
        if return_mask:
            return im_mask
        t1 = time.time()
        out_image = cv2.bitwise_and(image_array, image_array, mask=im_mask)
        # print("Time taken to bitwise and: ", time.time() - t1)
        return out_image


    def __predict__(self, file: sieve.File) -> sieve.File:
        import cv2
        import os
        import time

        video_extensions = ['mp4', 'avi', 'mov', 'flv']
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        file_extension = os.path.splitext(file.path)[1][1:]

        if file_extension in video_extensions:
            # check if output.mp4 or output1.mp4 exists and delete them
            if os.path.exists('output.mp4'):
                os.remove('output.mp4')
            if os.path.exists('output1.mp4'):
                os.remove('output1.mp4')
            cap = cv2.VideoCapture(file.path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
            count = 0
            while True:
                t = time.time()
                ret, frame = cap.read()
                if ret:
                    frame = self.__process_frame__(frame, return_mask = False)
                    out.write(frame)
                    print(f"Frame {count} / {num_frames} processed in {time.time() - t} seconds")
                else:
                    break
                count += 1
            cap.release()
            out.release()
            # add audio to output video with ffmpeg and subprocess
            import subprocess
            import os
            # get full path to current directory
            dir_path = os.path.dirname(os.path.realpath(__file__))
            command = ["ffmpeg", "-i", file.path, "-i", os.path.join(dir_path, "output.mp4"), "-map", "0:a", "-map", "1:v", "-c:v", "libx264", "-c:a", "aac", "output1.mp4"]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running subprocess: {e}")
            return sieve.Video(path='output1.mp4')
        elif file_extension in image_extensions:
            image = cv2.imread(file.path)
            image = self.__process_frame__(image, return_mask = False)
            cv2.imwrite('output.png', image)
            return sieve.Image(path='output.png')
        else:
            raise Exception("Invalid file type")
