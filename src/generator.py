import tensorflow as tf
import numpy as np
import glob
import h5py

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class DataGenerator():

    def __init__(self, directory, epochs=1, batch_size=1):
        super().__init__()
        self.directory = directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = int(
            len(glob.glob(self.directory+'*.mat')) / batch_size)
        self.stream = self.pipeline()

    def pipeline(self):
        train_data = tf.data.Dataset.from_generator(self.generator, output_types=(
            tf.float32, tf.uint8), output_shapes=(tf.TensorShape([512, 512, 1]), tf.TensorShape([512, 512, 1])))

        train_data = train_data.map(
            self.preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data = train_data.batch(self.batch_size)
        train_data = train_data.repeat(self.epochs)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
        return train_data

    def generator(self):
        while True:
            for filename in glob.glob(self.directory+'*.mat'):
                data = {}
                with h5py.File(filename, 'r') as file:
                    for struct in file:
                        attributes = file[struct]
                        for field in attributes:
                            if isinstance(field, str):
                                data[field] = attributes[field][()]
                            else:
                                data[struct] = field

                data['image'] = tf.expand_dims(data['image'], axis=-1)
                data['tumorMask'] = tf.expand_dims(data['tumorMask'], axis=-1)

                if data['image'].shape != (512, 512):
                    data['image'] = tf.image.resize(data['image'], (512, 512))
                    data['tumorMask'] = tf.image.resize(
                        data['tumorMask'], (512, 512))

                yield(data['image'], data['tumorMask'])

    def preprocess_fn(self, X_data, Y_data):
        """ Preprocesing of the data, i.e. normalize, resize and flip orientation of image """
        # Normalize image data to [0,1]
        X_data = tf.math.divide(tf.math.subtract(X_data, tf.math.reduce_min(
            X_data)), tf.math.subtract(tf.math.reduce_max(X_data), tf.math.reduce_min(X_data)))
        Y_data = tf.math.divide(tf.math.subtract(Y_data, tf.math.reduce_min(
            Y_data)), tf.math.subtract(tf.math.reduce_max(Y_data), tf.math.reduce_min(Y_data)))

        return X_data, Y_data
