import sieve

metadata = sieve.Metadata(
    title="DIS Background Removal",
    description="Highly Accurate Dichotomous Image Segmentation for Background Removal",
    code_url="https://github.com/sieve-community/examples/tree/main/background_removal",
    image=sieve.Image(
        url="https://github.com/xuebinqin/DIS/raw/main/figures/dis5k-v1-sailship.jpeg"
    ),
    tags=["Image", "Background", "Removal"],
    readme=open("DIS_README.md", "r").read(),
)

@sieve.Model(
    name="dis",
    gpu = True,
    python_packages=[
        "six==1.16.0",
        "pillow==9.3.0",
        "scikit-image==0.19.3",
        "torch==1.13.1",
        "torchvision==0.14.1",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/dis/models/",
        "wget -c 'https://storage.googleapis.com/mango-public-models/isnet-general-use.pth' -P /root/.cache/dis/models/"
    ],
    metadata=metadata
)
class Dis:
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
    
    def __predict__(self, input_image: sieve.Image) -> sieve.Image:
        """
        :param input_image: image to remove background from
        :return: image with background removed and mask
        """
        import numpy as np
        import cv2
        from skimage import io

        import torch
        from torchvision.transforms.functional import normalize
        import torch.nn.functional as F
        import uuid
        import time

        t1 = time.time()
        im = io.imread(input_image.path)
        # print("Time taken to read image: ", time.time() - t1)
        t1 = time.time()
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_shp=im.shape[0:2]
        # print("Time taken to check image shape: ", time.time() - t1)
        t1 = time.time()
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
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
        out_image = cv2.bitwise_and(input_image.array, input_image.array, mask=im_mask)
        yield sieve.Image(out_image)
        yield sieve.Image(array=im_mask)
