import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_chl, features_dim):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_chl, features_dim * 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(
                features_dim, features_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            self._block(
                features_dim * 2, features_dim * 4, kernel_size=4, stride=2, padding=1
            ),
            self._block(
                features_dim * 4, features_dim * 8, kernel_size=4, stride=2, padding=1
            ),
            nn.Cov2d(features_dim * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
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
            nn.Conv2d(in_chl, out_chl, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_chl) if norm else None,
            nn.LeakyReLU(0.2, inplace=True) if activation else None,
        )
        return out

    def forward(self, x):
        return self.dis(x)
