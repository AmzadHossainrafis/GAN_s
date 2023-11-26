# import torch tramsform
import torch
from torchvision.transforms import transforms
from DAN_s.components.utils.common import read_yaml


config = read_yaml("config/config.yaml") # read config.yaml
pre_process = config["Pre_process"] # get pre_process from config.yaml


transform = transforms.Compose([transforms.Resize(pre_process['image_size']), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5 for _ in range(pre_process['in_channel'])],\
                                                     [0.5 for _ in range(pre_process['in_channel'])]
                                                     )]) # transform image to tensor and normalize it to [-1,1




