"""
Utils model for GAN project
"""
import argparse
import os
import logging
import json
import random
from enum import Enum
import inspect

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
from torchvision.utils import save_image
from torchsummary import summary


REAL_LABEL = 1
FAKE_LABEL = 0

class Policy(Enum):
    """
    Batch size policy enum
    """
    SAME = 1
    FAKE_INCREASE = 2
    REAL_INCREASE = 3
    BOTH_INCREASE = 4

POLICIES = [policy.name.lower() for policy in Policy]

LOGGER = logging.getLogger(__file__)
LOGGER.setLevel(10)

def get_dataloader(opt, batch_size):
    """Returns a dataloader according to the paramters in opt

    Arguments:
        opt {[argparse.Namespace]} -- [program parmeters]
        batch_size {[int]} -- [batch size]

    Returns:
        [dataloader] -- [dataloader according to opt paramters with batches of size batch_size ]
    """

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
        dataset = datasets.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif opt.dataset == 'lsun':
        classes = [c + '_train' for c in opt.classes.split(',')]
        dataset = datasets.LSUN(
            root=opt.dataroot, classes=classes,
            transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.CenterCrop(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            root=opt.dataroot, download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif opt.dataset == 'mnist':
        dataset = datasets.MNIST(
            root=opt.dataroot, download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
    elif opt.dataset == 'fashionmnist':
        dataset = datasets.FashionMNIST(
            root=opt.dataroot, download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
    elif opt.dataset == 'fake':
        dataset = datasets.FakeData(
            image_size=(3, opt.image_size, opt.image_size),
            transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(opt.n_cpu))
    return dataloader

def get_number_channels(dataset):
    """ Returns the number of channels according to the dataset name
        Returns -1 if the dataset name is wrong

    Arguments:
        dataset {[string]} -- [dataset name]

    Returns:
        [int] -- [number of channels in the dataset]
    """
    if dataset in ['fake', 'cifar10', 'imagenet', 'folder', 'lfw', 'lsun']:
        return 3
    if dataset in ['mnist', 'fashionmnist']:
        return 1
    return -1

def get_batch_sizes(opt, epoch):
    """
    Return batch size of real samples and batch size of fake samples
    according to policy and epoch

    Arguments:
        opt {[argparse.Namespace]} -- [program parmeters]
        epoch {[int]} -- [epoch]

    Returns:
        [tuple] -- [(real batch size, fake batch size)]
    """
    policy = opt.policy.upper()
    if policy == Policy.SAME.name:
        return opt.batch_size, opt.batch_size
    if policy == Policy.FAKE_INCREASE.name:
        fake_batch_param = np.power(2, int(epoch/opt.batch_interval))
        return opt.batch_size, int(opt.batch_size * fake_batch_param)
    if policy == Policy.REAL_INCREASE.name:
        real_batch_param = np.power(2, int(epoch/opt.batch_interval))
        return int(opt.batch_size * real_batch_param), opt.batch_size
    if policy == Policy.BOTH_INCREASE.name:
        batch_param = np.power(2, int(epoch/opt.batch_interval))
        batch_size = int(opt.batch_size * batch_param)
        return batch_size, batch_size

def get_basic_parser():
    """ Basic argumnet parser

    Returns:
        [argparse.Namespace] -- [Basic argument parser]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist', help="Dataset to use",
                        choices=['cifar10', 'lsun', 'mnist',
                                 'imagenet', 'folder', 'lfw', 'fake', 'fashionmnist'])
    parser.add_argument('--dataroot', default="data", help='path to dataset')
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--image_size', type=int,
                        default=64, help='the height / width of the input image to network')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs of training')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='adam: decay of first order momentum of gradient, default=0.5')
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="adam: decay of moving average of squared gradient, default=0.999")
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom',
                        help='comma separated list of classes for the lsun data set')
    parser.add_argument("--policy", type=str, default='same',
                        help="Batch size policy",
                        choices=POLICIES)
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval betwen image samples")
    parser.add_argument("--batch_interval", type=int, default=25,
                        help="Intervals to update batch size in")
    parser.add_argument("--output_folder", type=str, default="images",
                        help="Output folder")
    parser.add_argument("--action", type=str, default="train",
                        help="Action - train model or print summary", choices=["train", "summary"])
    parser.add_argument("--device", type=str, help="device string")
    parser.add_argument('--net_g', default='', help="path to netG (to continue training)")
    parser.add_argument('--net_d', default='', help="path to netD (to continue training)")
    return parser

def get_device(opt):
    """ Return device type string to use

    Arguments:
        opt {[argparse.Namespace]} -- [program parmeters]

    Returns:
        [string] -- [device ype strign]
    """
    if opt.device:
        return opt.device
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def get_log_file_name(opt, orgin_file):
    """[Returns name of a log file according to the program parameters]

    Arguments:
        opt {[argparse.Namespace]} -- [Program arguemnts]
        orgin_file {[string]} -- [name of the file the function was called from]

    Returns:
        [string] -- [log file name]
    """
    os.makedirs("logs", exist_ok=True)
    return os.path.join(
        "logs", "{}_{}_{}_{}_{}.log".format(
            orgin_file.rsplit(".", 1)[0],
            opt.policy,
            opt.batch_interval,
            opt.batch_size, opt.dataset))

def train(opt, generator, discriminator,
          adversarial_loss,
          optimizer_g, optimizer_d,
          noise_dim, output_path):
    """ Orchestrate training

    Arguments:
        opt {[argparse.Namespace]} -- [training arguments]
        generator {[nn.Module]} -- [Generator network]
        discriminator {[nn.Module]} -- [Discriminator network]
        adversarial_loss {[type]} -- [GAN loss function]
        optimizer_g {[type]} -- [Optimizer for the generator network]
        optimizer_d {[type]} -- [Optimizer for the discriminator network]
        noise_dim {[tuple]} --  [Dimensions of generated noise without batch size]
        output_path {[string]} --  [Location to dump examples to generated images to]

    """
    batches_done = 0
    for epoch in range(opt.n_epochs):
        real_batch_size, fake_batch_size = get_batch_sizes(opt, epoch)
        dataloader = get_dataloader(opt, real_batch_size)
        for i, (data, _) in enumerate(dataloader):
            gen_imgs, d_loss, g_loss = \
                run_step(opt, data, fake_batch_size,
                         generator, discriminator,
                         adversarial_loss,
                         optimizer_g, optimizer_d,
                         noise_dim=noise_dim)
            if i%100 == 0:
                LOGGER.info(
                    json.dumps({"epoch": epoch,
                                "num_epochs": opt.n_epochs,
                                "batch": i,
                                "num_batches": len(dataloader),
                                "generator_loss": g_loss,
                                "discriminator_loss": d_loss,
                                "real_batch_size": real_batch_size,
                                "fake_batch_size": fake_batch_size}))
            batches_done += 1
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25],
                           os.path.join(output_path, "{}.png".format(batches_done)),
                           nrow=5, normalize=True)

def run_step(opt, data,
             fake_batch_size,
             generator, discriminator,
             adversarial_loss,
             optimizer_g, optimizer_d,
             noise_dim):
    """ Run a single training step

    Arguments:
        opt {[argparse.Namespace]} -- [training arguments]
        data {[Tensor]} -- [current batch of real data]
        fake_batch_size {[int]} -- [Number of fake examples to generates ]
        generator {[nn.Module]} -- [Generator network]
        discriminator {[nn.Module]} -- [Discriminator network]
        adversarial_loss {[type]} -- [GAN loss function]
        optimizer_g {[type]} -- [Optimizer for the generator network]
        optimizer_d {[type]} -- [Optimizer for the discriminator network]
        noise_dim {[tuple]} --  [Dimensions of generated noise without batch size]

    Returns:
        [tuple] -- [generated images, generator loss, discriminator loss]
    """
    data = data.to(opt.device)

    # Sample noise as generator input
    real_noise_dim = tuple([fake_batch_size] + list(noise_dim))
    noise = torch.randn(size=real_noise_dim, device=opt.device)
    # Generate a batch of images
    gen_imgs = generator(noise)
    # -----------------
    #  Train Generator
    # -----------------

    g_loss = train_generator(
        opt, discriminator,
        adversarial_loss, optimizer_g,
        gen_imgs)

    # ---------------------
    #  Train Discriminator
    # ---------------------

    d_loss = train_discriminator(
        opt, discriminator,
        adversarial_loss, optimizer_d,
        data, gen_imgs)

    return gen_imgs, d_loss.item(), g_loss.item()

def train_generator(opt,
                    discriminator,
                    adversarial_loss,
                    optimizer_g,
                    gen_imgs):
    """ Trains generator for one step

    Arguments:
        opt {[argparse.Namespace]} -- [training arguments]
        discriminator {[nn.Module]} -- [Discriminator network]
        adversarial_loss {[loss function]} -- [GAN loss function]
        optimizer_g {[type]} -- [Optimizer for the generator network]
        gen_imgs {[type]} -- [current batch of fake data]

    Returns:
        [float] -- [generator loss]
    """
    fake_batch_size = gen_imgs.shape[0]

    optimizer_g.zero_grad()

    # Loss measures generator's ability to fool the discriminator
    labels = torch.full((fake_batch_size,), REAL_LABEL, device=opt.device, requires_grad=False)
    output = discriminator(gen_imgs)
    g_loss = adversarial_loss(output, labels)
    g_loss.backward()
    # d_g_z2 = output.mean().item()
    optimizer_g.step()

    return g_loss

def train_discriminator(opt, discriminator,
                        adversarial_loss,
                        optimizer_d,
                        data, gen_imgs):
    """ Trains discriminator for one step

    Arguments:
        opt {[argparse.Namespace]} -- [training arguments]
        discriminator {[nn.Module]} -- [Discriminator network]
        adversarial_loss {[loss function]} -- [GAN loss function]
        optimizer_d {[type]} -- [Optimizer for the discriminator network]
        data {[Tensor]} -- [current batch of real data]
        gen_imgs {[type]} -- [current batch of fake data]

    Returns:
        [float] -- [discriminator loss]
    """

    optimizer_d.zero_grad()

    fake_batch_size = gen_imgs.shape[0]
    labels = torch.normal(
        mean=REAL_LABEL,
        std=0.10,
        size=(data.size(0),),
        requires_grad=False,
        device=opt.device)
    output = discriminator(data)
    real_loss = adversarial_loss(output, labels)
    real_loss.backward()

    labels = torch.normal(
        mean=FAKE_LABEL,
        std=0.10,
        size=(fake_batch_size,),
        requires_grad=False,
        device=opt.device)
    output = discriminator(gen_imgs.detach())
    fake_loss = adversarial_loss(output, labels)
    fake_loss.backward()

    d_loss = real_loss + fake_loss
    optimizer_d.step()
    return d_loss


def run_train_session(opt, generator, discriminator,
                      noise_dim, origin_file):
    """ Run full training session

    Arguments:
        opt {[type]} -- [description]
        generator {[type]} -- [description]
        discriminator {[type]} -- [description]
        noise_dim {[type]} -- [description]
        origin_file {[string]} -- [The file name which the call was originated from]
    """
    if opt.manual_seed is None:
        opt.manual_seed = random.randint(1, 10000)
    #LOGGER.info("Random Seed: {}".format(opt.manual_seed))
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    # Load models
    if opt.net_d != '':
        discriminator.load_state_dict(torch.load(opt.net_d))
    if opt.net_g != '':
        generator.load_state_dict(torch.load(opt.net_g))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=opt.lr, betas=(opt.beta1, opt.beta2))

    output_path = os.path.join(opt.output_folder,
                               opt.dataset,
                               origin_file.rsplit(".")[0],
                               "{}_{}_{}".format(opt.policy,
                                                 opt.batch_interval,
                                                 opt.batch_size))
    os.makedirs(output_path, exist_ok=True)

    train(
        opt,
        generator,
        discriminator,
        adversarial_loss,
        optimizer_g,
        optimizer_d,
        noise_dim=noise_dim,
        output_path=output_path)

def get_image_shape(dataset, image_size):
    """ Return image shape

    Arguments:
        datasets {[string]} -- [dataset name]
        image_size {[int]} -- [image height and width (assumes rectangle image)]

    Returns:
        [tuple] -- [image dimensions, (channel, height, width)]
    """
    channels = get_number_channels(dataset)
    img_shape = (channels, image_size, image_size)
    return img_shape

def model_summary(opt, generator, discriminator, noise_dim):
    """ Prints fgenerator and discriminator summaries

    Arguments:
        opt {[argparse.Namespace]} -- [program arguments]
        generator {[nn.Module]} -- [Generator network]
        discriminator {[nn.Module]} -- [Discriminator network]
        noise_dim {[tuple]} --  [Dimensions of generated noise without batch size]
    """
    img_shape = get_image_shape(opt.dataset, opt.image_size)
    summary(generator, noise_dim)
    summary(discriminator, img_shape)

def set_logging(opt, origin_file):
    """Creates a log file

    Arguments:
        opt {[argparse.Namespace]} -- [program arguments]
        origin_file {[string]} -- [The file name which the call was originated from]
    """
    file_logger = logging.FileHandler(get_log_file_name(opt, origin_file), mode='w')
    formatter = logging.Formatter('%(message)s')
    file_logger.setFormatter(formatter)

    LOGGER.addHandler(file_logger)

    LOGGER.info("{}".format(opt))

def run_program(opt, generator, discriminator, noise_dim):
    """ Run program according to action option

    Arguments:
        opt {[argparse.Namespace]} -- [program arguments]
        generator {[nn.Module]} -- [Generator network]
        discriminator {[nn.Module]} -- [Discriminator network]
        noise_dim {[tuple]} --  [Dimensions of generated noise without batch size]
    """
    origin_file = inspect.stack()[1].filename
    set_logging(opt, origin_file)

    if opt.action == "summary":
        model_summary(
            opt,
            generator,
            discriminator,
            noise_dim)
    else:
        run_train_session(
            opt,
            generator,
            discriminator,
            noise_dim,
            origin_file)
