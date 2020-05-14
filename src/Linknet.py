import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# source ./venv/bin/activate

print(('TensorFlow version: {}').format(tf.__version__))

class LinkNet:

    def __init__(self, image_shape, num_classes, learning_rate = 0.001):
        self.image_shape = image_shape
        self.conv_size = 3
        self.deconv_size = 3
        self.activation = 'relu'
        self.num_outputs = num_classes if num_classes > 2 else 1
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = self.build_model()

    def predict(self, input):
        predicted = self.model.predict(input)
        return predicted

    def fit(samples, targets, batch_size, epochs, save_model = False, checkpoint_path = None, save_period = 1):
        if save_model:
            self.checkpoint_path = checkpoint_path
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                                     save_weights_only=True,
                                                                     verbose=1,
                                                                     period = save_period)
            callbacks = [checkpoint_callback]
        else:
            callbacks = None

        self.model.fit(samples, targets, batch_size, epochs, callbacks = callbacks)

    def train(self):
        """Write a custom training loop."""
        pass

    def load_latest_model(self):
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.model = self.build_model()
        self.model.load_weights(latest)

    def build_model(self):

        # Initial block
        input = tf.keras.Input(shape = self.image_shape)
        init = self.add_input_block(input, nf = 64, pooling_size = 3)

        enc1 = self.add_encoder_block(init, nf = 64)
        enc2 = self.add_encoder_block(enc1, nf = 128)
        enc3 = self.add_encoder_block(enc2, nf = 256)
        enc4 = self.add_encoder_block(enc3, nf = 512)

        dec4 = self.add_decoder_block(enc4, nf = 256)
        dec3 = tf.keras.layers.add([dec4, enc3])
        dec3 = self.add_decoder_block(dec3, nf = 128)

        dec2 = tf.keras.layers.add([dec3, enc2])
        dec2 = self.add_decoder_block(dec2, nf = 64)

        dec1 = tf.keras.layers.add([dec2, enc1])
        dec1 = self.add_decoder_block(dec2, nf = 64)
        output = self.add_output_block(dec1, nf = 32)

        model = tf.keras.Model(input, output)
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics=['accuracy'])
        return model


    def add_input_block(self, input, nf, pooling_size, apply_batchnorm = True):

        init = tf.keras.layers.Conv2D(nf, kernel_size = 7, strides = 2, padding = "same")(input)

        if apply_batchnorm:
            init = tf.keras.layers.BatchNormalization()(init)

        init = tf.keras.layers.Activation(self.activation)(init)
        init = tf.keras.layers.MaxPool2D(pool_size = pooling_size, strides = 2)(init)

        return init

    def add_output_block(self, input, nf, apply_batchnorm = True):
        l1 = tf.keras.layers.Conv2DTranspose(nf, self.deconv_size, strides = 2, padding = "same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2D(nf, self.conv_size, padding = "same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)
        l2 = tf.keras.layers.Activation(self.activation)(l2)

        l3 = tf.keras.layers.Conv2DTranspose(self.num_outputs, 2, strides = 2, padding = "same")(l2)
        if apply_batchnorm:
            l3 = tf.keras.layers.BatchNormalization()(l3)

        if self.num_outputs > 1:
            output_activation = "softmax"
        else:
            output_activation = "sigmoid"
        l3 = tf.keras.layers.Activation(output_activation)(l3)
        return l3

    def add_encoder_block(self, input, nf, apply_batchnorm = True):
        res1 = self.add_residual_block(input, nf, downsample = True, apply_batchnorm = apply_batchnorm)
        res2 = self.add_residual_block(res1, nf, downsample = False, apply_batchnorm = apply_batchnorm)
        return res2

    def add_decoder_block(self, input, nf, apply_batchnorm = True):
        input_shape = input.shape
        m = input_shape[-1] # Channels last, right?

        l1 = tf.keras.layers.Conv2D(m / 4, kernel_size = 1, padding = "same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2DTranspose(m / 4, kernel_size = self.deconv_size, strides = 2, padding = "same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)
        l2 = tf.keras.layers.Activation(self.activation)(l2)

        l3 = tf.keras.layers.Conv2D(nf, kernel_size = 1, padding = "same")(l2)
        if apply_batchnorm:
            l3 = tf.keras.layers.BatchNormalization()(l3)
        l3 = tf.keras.layers.Activation(self.activation)(l3)

        return l3

    def add_residual_block(self, input, nf, downsample, apply_batchnorm = True):

        stride_length = 2 if downsample else 1

        l1 = tf.keras.layers.Conv2D(nf, self.conv_size, strides = stride_length, padding = "same")(input)
        if apply_batchnorm:
            l1 = tf.keras.layers.BatchNormalization()(l1)
        l1 = tf.keras.layers.Activation(self.activation)(l1)

        l2 = tf.keras.layers.Conv2D(nf, self.conv_size, padding = "same")(l1)
        if apply_batchnorm:
            l2 = tf.keras.layers.BatchNormalization()(l2)

        if downsample:
            shortcut = tf.keras.layers.Conv2D(nf, kernel_size = 1, strides = 2, padding = "same")(input)
        else:
            shortcut = input

        output = tf.keras.layers.add([l2, shortcut])
        output = tf.keras.layers.Activation(self.activation)(output)

        return output

