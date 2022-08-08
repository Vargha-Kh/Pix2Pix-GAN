from __future__ import print_function, division
import scipy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, BatchNormalization, \
    Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard


class Pix2Pix():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.G_losses = []
        self.D_losses = []
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.generator = self.build_generator()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A = self.generator(img_B)

        self.discriminator.trainable = False

        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))

    def encoder_block(self, layer_input, num_filters, batchnorm=True):
        initializers = RandomNormal(stddev=0.02)
        x = Conv2D(num_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializers)(layer_input)
        x = LeakyReLU(alpha=0.2)(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x

    def decoder_block(self, layer_input, skip_input, num_filters, dropout_rate=0):
        initializers = RandomNormal(stddev=0.02)
        up = UpSampling2D(size=2)(layer_input)
        x = Conv2D(num_filters, (4, 4), strides=1, padding='same', kernel_initializer=initializers, activation='relu')(
            up)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        x = Concatenate()([x, skip_input])
        return x

    def build_generator(self):
        initializers = RandomNormal(stddev=0.02)
        input_image = Input(shape=self.img_shape)
        e1 = self.encoder_block(input_image, 64, batchnorm=False)
        e2 = self.encoder_block(e1, 128)
        e3 = self.encoder_block(e2, 256)
        e4 = self.encoder_block(e3, 512)
        e5 = self.encoder_block(e4, 512)
        e6 = self.encoder_block(e5, 512)
        e7 = self.encoder_block(e6, 512)

        d2 = self.decoder_block(e7, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512)
        d5 = self.decoder_block(d4, e3, 256)
        d6 = self.decoder_block(d5, e2, 128)
        d7 = self.decoder_block(d6, e1, 64)
        up = UpSampling2D(size=2)(d7)
        output_image = Conv2D(self.channels, (4, 4), strides=1, padding='same', kernel_initializer=initializers,
                              activation='tanh')(up)
        model = Model(input_image, output_image)
        return model

    def build_discriminator(self):
        initializers = RandomNormal(stddev=0.02)
        input_source_image = Input(self.img_shape)
        input_target_image = Input(self.img_shape)
        merged_input = Concatenate(axis=-1)([input_source_image, input_target_image])
        filters_list = [64, 128, 256, 512]

        def disc_layer(input_layer, filters, kernel_size=(4, 4), batchnorm=True):
            x = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same', kernel_initializer=initializers)(
                input_layer)
            x = LeakyReLU(0.2)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            return x

        x = disc_layer(merged_input, filters_list[0], batchnorm=False)
        x = disc_layer(x, filters_list[1])
        x = disc_layer(x, filters_list[2])
        x = disc_layer(x, filters_list[3])

        discriminator_output = Conv2D(1, kernel_size=(4, 4), padding='same', kernel_initializer=initializers)(x)
        model = Model([input_source_image, input_target_image], discriminator_output)
        return model

    def train(self, epochs, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                fake_A = self.generator.predict(imgs_B)

                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f,D Loss Real: %f, D Loss Fake: %f, acc: %3d%%] [G loss: %f, L1 Loss: %f, acc: %3d%%]" % (
                    epoch, epochs,
                    batch_i, self.data_loader.n_batches,
                    d_loss[0], d_loss_real[0], d_loss_fake[0], 100 * d_loss[1],
                    g_loss[0], g_loss[1], 100 * g_loss[2]))
                self.G_losses.append(g_loss[0])
                self.D_losses.append(d_loss[0])
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    self.plot_metrics(self.G_losses, self.D_losses)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def plot_metrics(self, G_losses, D_losses):
        plt.figure(figsize=(8, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
