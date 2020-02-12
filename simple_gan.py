"""
Simple GAN training
"""

import sys

import numpy as np

import torch.nn as nn

import utils

def parse_arguments(args):
    """ Parse program arguments

    Arguments:
        args {[string]} -- Arguments to parse

    Returns:
        [argparse.Namespace] -- parsed arguments
    """
    parser = utils.get_basic_parser()
    parser.usage = "MNIST GAN"
    parser.description = "MNIST GAN"
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    opt = parser.parse_args(args)
    opt.device = utils.get_device(opt)
    return opt

class Generator(nn.Module):
    """
    Generator class
    """
    def __init__(self, opt, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """
    Discriminator class
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        output = self.model(img_flat)
        return output.view(-1, 1).squeeze(1)

def main(args):
    """
    Run training process
    """
    opt = parse_arguments(args)

    # Initialize generator and discriminator
    channels = utils.get_number_channels(opt.dataset)
    img_shape = (channels, opt.image_size, opt.image_size)

    generator = Generator(opt, img_shape).to(opt.device)
    discriminator = Discriminator(img_shape).to(opt.device)
    noise_dim=(opt.latent_dim,)
    utils.run_program(opt, generator, discriminator, noise_dim)

if __name__ == "__main__":
    main(sys.argv[1:])
