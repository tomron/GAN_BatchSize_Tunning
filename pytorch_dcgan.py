"""
Train GAN  using conv and deconv networks on multiple possible dataset
"""

from __future__ import print_function
import sys

import torch.nn as nn

import utils

def parse_arguments(args):
    """
    Parse program arguments
    """
    parser = utils.get_basic_parser()
    parser.usage = "DCGAN"
    parser.description = "DCGAN"
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='the depth of feature maps carried through the generator')
    parser.add_argument('--ndf', type=int, default=64,
                        help='the depth of feature maps propagated through the discriminator')
    opt = parser.parse_args(args)
    opt.device = utils.get_device(opt)

    return opt

def weights_init(m):
    """
    Custom weights initialization called on generator and discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    Generator class
    """
    def __init__(self, nz, nc, ngf, n_gpu):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(negative_slope=0.2),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(negative_slope=0.2),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(negative_slope=0.2),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        if x.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.n_gpu))
        else:
            output = self.main(x)
        return output


class Discriminator(nn.Module):
    """
    Discriminator class
    """
    def __init__(self, nc, ndf, n_gpu):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.is_cuda and self.n_gpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.n_gpu))
        else:
            output = self.main(x)

        return output.view(-1, 1).squeeze(1)

def main(args):
    """
    Main
    """
    opt = parse_arguments(args)

    # Initialize generator and discriminator
    n_channels = utils.get_number_channels(opt.dataset)

    generator = Generator(nz=opt.nz, nc=n_channels,
                          ngf=opt.ngf, n_gpu=opt.n_gpu).to(opt.device)
    generator.apply(weights_init)

    discriminator = Discriminator(nc=n_channels, ndf=opt.ndf,
                                  n_gpu=opt.n_gpu).to(opt.device)
    discriminator.apply(weights_init)
    noise_dim = (opt.nz, 1, 1)
    utils.run_program(opt, generator, discriminator, noise_dim)

if __name__ == "__main__":
    main(sys.argv[1:])
