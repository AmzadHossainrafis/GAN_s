import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_chl, features_dim):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._block(in_chl, features_dim * 16, kernel_size=4, stride=2, padding=1),
            self._block(
                features_dim * 16, features_dim * 8, kernel_size=4, stride=2, padding=1
            ),
            self._block(
                features_dim * 8, features_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            self._block(
                features_dim * 4, features_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ConvTranspose2d(features_dim * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(
        self,
        in_chl,
        out_chl,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False,
        norm=True,
        activation=True,
    ):
        out = nn.Sequential(
            nn.ConvTranspose2d(
                in_chl, out_chl, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_chl) if norm else None,
            nn.ReLU(inplace=True) if activation else None,
        )
        return out

    def forward(self, x):
        return self.gen(x)
