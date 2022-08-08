import os
import numpy as np
import math
import itertools
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import time
import datetime
from models import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

G_losses = []
D_losses = []
def training(data_dir, init_epoch=0, n_epochs=200, dataset_name="facades", batch_size=1, img_width=256, img_height=256,
             lr=0.0002,
             b1=0.5, b2=0.999, num_workers=8, lambda_pixel=100, sample_interval=500, checkpoint_interval=1):
    os.makedirs("images/%s" % dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if init_epoch != 0:
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, init_epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, init_epoch)))
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset(data_dir + "%s" % dataset_name, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        ImageDataset(data_dir + "%s" % dataset_name, transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_images(batches_done):
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["B"].type(Tensor))
        real_B = Variable(imgs["A"].type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

    def plot_metrics(G_losses, D_losses):
        plt.figure(figsize=(8, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    #  Training
    prev_time = time.time()
    for epoch in range(init_epoch, n_epochs):
        for i, batch in enumerate(dataloader):

            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))

            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            #  Train Generators
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            batches_left = n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), loss_pixel.item(),
                   loss_GAN.item(), time_left)
            )
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
                plot_metrics(G_losses, D_losses)

        if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))


if __name__ == "__main__":
    path = "/home/vargha/Desktop/pix2pix_pytorch/datasets/"
    training(data_dir=path)
