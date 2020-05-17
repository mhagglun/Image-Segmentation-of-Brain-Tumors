import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util import dice_score
import os
import datetime


class LinkNet:

    def __init__(self, image_shape, num_classes, learning_rate=1e-3):
        self.image_shape = image_shape
        self.conv_size = 3
        self.deconv_size = 3
        self.activation = 'relu'
        self.initialization = 'he_normal'
        self.regularization = 1e-4
        self.num_outputs = num_classes if num_classes > 2 else 1
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.model = self.build_model()

    def build_model(self):

        # Initial block
        input = tf.keras.Input(shape=self.image_shape)
        init = self.add_input_block(input, nf=64, pooling_size=3)

        enc1 = self.add_encoder_block(init, nf=64, apply_batchnorm=True)
        enc2 = self.add_encoder_block(enc1, nf=128, apply_batchnorm=True)
        enc3 = self.add_encoder_block(enc2, nf=256, apply_batchnorm=True)
        enc4 = self.add_encoder_block(enc3, nf=512, apply_batchnorm=True)

        dec4 = self.add_decoder_block(enc4, nf=256, apply_batchnorm=True)
        dec3 = tf.keras.layers.add([dec4, enc3])
        dec3 = self.add_decoder_block(dec3, nf=128, apply_batchnorm=True)

        dec2 = tf.keras.layers.add([dec3, enc2])
        dec2 = self.add_decoder_block(dec2, nf=64, apply_batchnorm=True)

        dec1 = tf.keras.layers.add([dec2, enc1])
        dec1 = self.add_decoder_block(dec1, nf=64, apply_batchnorm=True)
        output = self.add_output_block(dec1, nf=32, apply_batchnorm=True)

        model = tf.keras.Model(input, output, name='Linknet')
        model.compile(optimizer=self.optimizer, loss=self.loss,
                      metrics=['accuracy', dice_score])
        return model

    def add_input_block(self, input, nf, pooling_size, apply_batchnorm=True):

        init = tf.keras.layers.Conv2D(
            nf, kernel_size=7, strides=2, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(input)

        if apply_batchnorm:
            init = tf.keras.layers.BatchNormalization()(init)

        init = tf.keras.layers.Activation(self.activation)(init)
        init = tf.keras.layers.MaxPool2D(
            pool_size=pooling_size, strides=2)(init)

        return init

    def add_output_block(self, input, nf, apply_batchnorm=True):
        l1 = tf.keras.layers.Conv2DTranspose(
            nf, self.deconv_size, strides=2, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2D(nf, self.conv_size, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)
        l2 = tf.keras.layers.Activation(self.activation)(l2)

        l3 = tf.keras.layers.Conv2DTranspose(
            self.num_outputs, 2, strides=2, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(l2)
        if apply_batchnorm:
            l3 = tf.keras.layers.BatchNormalization()(l3)

        if self.num_outputs > 1:
            output_activation = "softmax"
        else:
            output_activation = "sigmoid"
        l3 = tf.keras.layers.Activation(output_activation)(l3)
        return l3

    def add_encoder_block(self, input, nf, apply_batchnorm=True):
        res1 = self.add_residual_block(
            input, nf, downsample=True, apply_batchnorm=apply_batchnorm)
        res2 = self.add_residual_block(
            res1, nf, downsample=False, apply_batchnorm=apply_batchnorm)
        return res2

    def add_decoder_block(self, input, nf, apply_batchnorm=True):
        input_shape = input.shape
        m = input_shape[-1]  # Channels last, right?

        l1 = tf.keras.layers.Conv2D(
            int(m / 4), kernel_size=1, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2DTranspose(
            int(m / 4), kernel_size=self.deconv_size, strides=2, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)
        l2 = tf.keras.layers.Activation(self.activation)(l2)

        l3 = tf.keras.layers.Conv2D(nf, kernel_size=1, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(l2)
        if apply_batchnorm:
            l3 = tf.keras.layers.BatchNormalization()(l3)
        l3 = tf.keras.layers.Activation(self.activation)(l3)

        return l3

    def add_residual_block(self, input, nf, downsample, apply_batchnorm=True):

        stride_length = 2 if downsample else 1

        l1 = tf.keras.layers.Conv2D(
            nf, self.conv_size, strides=stride_length, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2D(nf, self.conv_size,  kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)

        if downsample:
            shortcut = tf.keras.layers.Conv2D(
                nf, kernel_size=1, strides=2, kernel_initializer=self.initialization, kernel_regularizer = tf.keras.regularizers.l2(self.regularization), padding="same")(input)
        else:
            shortcut = input

        output = tf.keras.layers.add([l2, shortcut])
        output = tf.keras.layers.Activation(self.activation)(output)

        return output
