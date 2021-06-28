import torch
from model.FACNN import FACNN,LDSR_V4
from model.sam import SAM
from model.calculator import PSNR_tensor,SSIM,LPIPS
import torch.backends.cudnn as cudnn
from dataload.Image_dataset import TrainDataset
import yaml, os
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import wandb
import lpips as lp

class Main():

    def __init__(self,opt,wandbed):
        self.opt = opt
        if wandbed:
            wandb.init(project="FACNN")
            wandb.config.update(opt)
        self.setDevice()

    def setDevice(self):
        """ 환경 및 기기 설정 """
        # GPU 환경인 경우 CUDA 및 GPU 등록
        if self.opt["device"] == "gpu":
            # CUDA 환경 설정
            cudnn.deterministic = True
            cudnn.benchmark = True
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt["gpu_ids"])

            # GPU 등록 현재는 싱글만 가능 추후 멀티 프로세싱 추가 예정
            self.gpu_ids = [i for i in range(len(str(self.opt['gpu_ids']).split(",")))]
            self.device = torch.device(f"cuda:0")            
        # CPU 환경
        elif self.opt["device"] == "cpu":
            self.device = torch.device("cpu")
    def getModel(self):
        model = FACNN(self.opt["scale_factor"]).to(self.device)
        if os.path.exists(self.opt["model_pretrained_path"]):
            print("model loaded")
            model.load_state_dict(torch.load(self.opt["model_pretrained_path"]))
            return model
        else:
            return model
            

    def train(self):
        dataset = TrainDataset(self.opt,self.opt["train_dataset_name"],self.device)
        model = self.getModel()
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(),base_optimizer, lr=0.0001, betas=(0.9, 0.999))
        pixel_criterion = torch.nn.L1Loss().to(self.device)
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=8,
            shuffle=True,
            pin_memory=True,
            sampler=None,
            num_workers=8)
        img  = Image.open("image/80049126.jpg")
        origin = transforms.ToTensor()(img).to(self.device)
        lr = img.resize((img.width//2, img.height//2), Image.BICUBIC)
        base_image = transforms.ToTensor()(lr)
        base_image = base_image.unsqueeze(0)
        lpips_metric = lp.LPIPS(net='vgg')
        lpips_metric.to(self.device)

        best_psnr = 0
        best_ssim = 0
        best_lpips = 1
        for epoch in range(int(self.opt["epoch"])):
            with tqdm(total=len(dataloader), ncols=160) as t:
                for i, (lr, hr) in enumerate(dataloader):
                    model.train()
                    # 데이터를 장치에 등록
                    lr = lr.to(self.device)
                    hr = hr.to(self.device)

                    optimizer.zero_grad()

                    sr = model(lr)
                    loss = pixel_criterion(sr,hr)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    pixel_criterion(model(lr),hr).backward()
                    optimizer.second_step(zero_grad=True)

                    t.set_postfix(loss='{:.6f}'.format(loss))
                    t.update(1)
            with torch.no_grad():
                model.eval()
                base_image = base_image.to(self.device)
                result = model(base_image)
                vutils.save_image(result.detach(), "image/result.png")
                psnr = PSNR_tensor(origin,result)
                np_origin = origin.permute(1,2,0).cpu().detach().numpy()
                np_result = result.squeeze().permute(1,2,0).cpu().detach().numpy()
                ssim = SSIM(np_origin,np_result)
                lpips = LPIPS(np_origin,np_result,lpips_metric,self.device)
                if psnr >= best_psnr and ssim >= best_ssim and lpips<=best_lpips:
                    best_psnr = psnr
                    best_ssim = ssim
                    best_lpips = lpips
                    print(f"Best PSNR : {best_psnr} , Best SSIM : {best_ssim}, Best LPIPS : {best_lpips[0][0][0][0]}")
                    torch.save(model.state_dict(), "best.pth")
                save_epoch = "{0:04d}".format(epoch+275)
                torch.save(model.state_dict(), os.path.join("weights", f"x2/{save_epoch}.pth"))
                if opt["wandb"]:
                    wandb.log({
                    "SR_img": [wandb.Image(result.detach(), caption=epoch)],
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS":lpips,
                    "loss": loss})
        
if __name__ == '__main__':
    # 옵션 불러오기
    with open("FACNN.yml", mode="r", encoding="utf-8") as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    main = Main(opt,opt["wandb"])
    if opt["test"] == False:
        main.train()
    else:
        if os.path.exists(opt["model_pretrained_path"]):
            main.eval()
        else:
            raise Exception("Model path를 입력해주세요")