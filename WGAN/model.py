import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self,channel_img, features_d):
        super(Critic, self).__init__()
        self.cric = nn.Sequential(
           # input:
            nn.Conv2d(channel_img, features_d, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Output dimension is one
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        return self.cric(x)

class Genarator(nn.Module):
    def __init__(self, channel_noise, channels_img, features_g):
        super(Genarator, self).__init__()
        self.gen = nn.Sequential(
            self._block(channel_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    cric = Critic(in_channels, 8)
    # print(cric(x).size())
    assert cric(x).size() == (N,1,1,1), "Error in Critic"
    gen = Genarator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    # print(gen(z).size())
    assert gen(z).size() == (N, in_channels, H, W), "Error in Generator"

# test()