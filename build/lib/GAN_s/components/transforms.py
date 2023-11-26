# import torch tramsform
import torch
from torchvision.transforms import transforms
from DAN_s.components.utils.common import read_yaml


config = read_yaml("config.yaml") # read config.yaml


transform = transforms.Compose([transforms.Resize(), 
transforms.ToTensor(),
 transforms.Normalize()])




