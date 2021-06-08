import cv2
import numpy as np
import glob
import torch
from model.FACNN import FACNN
# from models.generator import Generator
import torchvision.transforms as transforms
# from models import ESPCN,HSDSR
# from utils import preprocess, calc_psnr , SSIM
from videoIO import VideoIO
import time
import os
# img = cv2.imread("./testsets/RealSRSet/S02_Movie_1080p_00017.png")
# height, width, layers = img.shape
# size = (width*2,height*2)
# dst = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("17.png",dst)
# img_array = []
# for filename in glob.glob('./testsets/RealSRSet/*'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     dst = cv2.resize(img, dsize=(width*2, height*2), interpolation=cv2.INTER_CUBIC)
#     img_array.append(dst)
 
 
# out = cv2.VideoWriter('BSRGAN_bicubic.mp4',cv2.VideoWriter_fourcc(*'X264'), 30, (width*2, height*2))
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
if __name__=="__main__":
    scale = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    # model = ESPCN(scale_factor=scale).to(device)

    # state_dict = model.state_dict()
    # for n, p in torch.load("weights/x2/3999.pth", map_location=lambda storage, loc: storage)['model_state_dict'].items():
    #     if n in state_dict.keys():
    #         state_dict[n].copy_(p)
    #     else:
    #         raise KeyError(n)
    
    model = FACNN(2).to(device)
    print(model)
    model.load_state_dict(torch.load("weights/x2_back/0135.pth"))
    model.eval()
    video_path = "S02_Movie_720p.ts"
    videoIO = VideoIO('sssss')
    videoIO.initVideoCapture(video_path)
    #initVideoWriter(self, width, height, target_video_path='test.mp4', fps=29.97):
    videoIO.initVideoWriter(2560,1440, 'test135.mp4')
    for _ in range(videoIO.getTotalFrame()):
        ret, frame = videoIO.getFrame()
        
        if ret:
            start = time.time()
            #전처리
            # lr = preprocess(frame, device)
            totensor = transforms.ToTensor()
            lr = totensor(frame).unsqueeze(0).to(device)
            print(f"preprocess : {time.time()-start}")
            bicubic = cv2.resize(frame, (2560,1440), interpolation=cv2.INTER_CUBIC)
            # lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)
            #SR
            start = time.time()
            with torch.no_grad():
                # sr = model(lr).clamp(0.0, 1.0)
                sr = model(lr)
            print(f"only SR time : {time.time()-start}")
            start = time.time()
            # sr = sr.squeeze(0).cpu().detach().numpy()
            sr = sr.cpu().numpy()
            # sr = np.clip(sr, 0.0, 255.0).astype(np.uint8)
            print(f"back process time : {time.time()-start}")
            output = videoIO.mergeFrame(bicubic,sr)
            # videoIO.showFrame(output)
            videoIO.saveVideo(output)