import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from GAN_s.components.models.dis import Discriminator
from GAN_s.components.models.gen import Generator
from GAN_s.components.utils.common import read_yaml
import torchvision
from transforms import transform 
import mlflow
import mlflow.pytorch

config = read_yaml("config/config.yaml")


class Trainer:
    def __init__(self, gen, dis) -> None:
        self.gen = gen
        self.dis = dis
        self.train_config = config["Train_config"]
        self.dis_opt = torch.optim.Adam(
            self.dis.parameters(), lr=self.train_config["lr"], betas=(0.5, 0.999)
        )
        self.gen_opt = torch.optim.Adam(
            self.gen.parameters(), lr=self.train_config["lr"], betas=(0.5, 0.999)
        )
        self.criterion = nn.BCELoss()  ### loss function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def inisialize_weight(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def train(self):
        self.gen.train()
        self.dis.train()
        for epoch in range(self.train_config["epoch"]):
            for i, (real_img, _) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader)
            ):
                real = real_img.to(self.device) 
                noise = torch.randn(real.size(0), self.train_config["z_dim"]).to(
                    self.device
                )
            dice_real= self.dis(real).reshape(-1)
            loss_dis_real = self.criterion(dice_real, torch.ones_like(dice_real))
            dice_fake = self.dis(self.gen(noise)).reshape(-1)
            loss_dis_fake = self.criterion(dice_fake, torch.zeros_like(dice_fake))
            loss_dis = (loss_dis_real + loss_dis_fake) / 2
            self.dis_opt.zero_grad()
            loss_dis.backward(retain_graph=True)
            self.dis_opt.step()
           

            output = self.dis(self.gen(noise)).reshape(-1)
            loss_gen = self.criterion(output, torch.ones_like(output))
            self.gen_opt.zero_grad()
            loss_gen.backward()
            self.gen_opt.step()

            if (i + 1) % self.train_config["log_step"] == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}".format(
                        epoch,
                        self.train_config["epoch"],
                        i + 1,
                        len(self.train_loader),
                        loss_dis.item(),
                        loss_gen.item(),
                    )
                )
            if (epoch + 1) % self.train_config["sample_step"] == 0:
                fake_images = self.gen(self.fixed_z)
                torchvision.utils.save_image(
                    fake_images,
                    f"outputs/fake_images-{epoch+1}.png",
                    normalize=True,
                )
                torchvision.utils.save_image(
                    real_img,
                    f"outputs/real_images-{epoch+1}.png",
                    normalize=True,
                )
     