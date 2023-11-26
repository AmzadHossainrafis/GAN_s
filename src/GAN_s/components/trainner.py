import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from GAN_s.components.models.dis import Discriminator
from GAN_s.components.models.gen import Generator
from GAN_s.components.utils.common import read_yaml
import mlflow
import mlflow.pytorch

config = read_yaml("../../config/config.yaml")


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
        pass
