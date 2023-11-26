import pytest # type: ignore
import torch 
import torch.nn as nn 
import numpy as np 
from GAN_s.components.models.dis import Discriminator   
from GAN_s.components.models.gen import Generator


def test_shape():
    """ Test shape of the output of the generator and the discriminator """
    # Set up the generator and the discriminator
    gen = Generator()
    dis = Discriminator()
    # Set up the input
    x = torch.randn(64, 100, 1, 1)
    # Get the output of the generator
    gen_out = gen(x)
    # Get the output of the discriminator
    dis_out = dis(gen_out)
    # Check the shape of the output of the generator
    assert gen_out.shape == (64, 1, 28, 28)
    # Check the shape of the output of the discriminator
    assert dis_out.shape == (64, 1, 1, 1)


    
    