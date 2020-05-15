import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import model
import h5py
import glob
import preprocessing

sns.set_style('darkgrid')

# tf.config.optimizer.set_jit(True)
# tf.debugging.set_log_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# 

# dataset = preprocessing.Dataset().load_batch('data/train_images')

batch_size = 1
epochs = 10
steps_per_epoch = int(len(glob.glob('data/braintumor/*.mat')) / batch_size)

steps_per_epoch_val = int(len(glob.glob('data/braintumorval/*.mat')) / batch_size)

def gen_file():
    while True:
        for filename in glob.glob('data/braintumor/*.mat'):
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
                data['tumorMask'] = tf.image.resize(data['tumorMask'], (512, 512))

            data['image'] = tf.image.convert_image_dtype(data['image'], tf.float32)
            data['tumorMask'] = tf.image.convert_image_dtype(data['tumorMask'], tf.uint8)

            yield(data['image'], data['tumorMask'])

def gen_file_val():
    while True:
        for filename in glob.glob('data/braintumorval/*.mat'):
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
                data['tumorMask'] = tf.image.resize(data['tumorMask'], (512, 512))

            data['image'] = tf.image.convert_image_dtype(data['image'], tf.float32)
            data['tumorMask'] = tf.image.convert_image_dtype(data['tumorMask'], tf.uint8)

            yield(data['image'], data['tumorMask'])


def preprocess_fn(X_data, Y_data):
        """ Preprocesing of the data, i.e. normalize, resize and flip orientation of image """
        # Normalize image data to [0,1]
        X_data = tf.math.divide(tf.math.subtract(X_data, tf.math.reduce_min(X_data)), tf.math.subtract(tf.math.reduce_max(X_data), tf.math.reduce_min(X_data)))
        Y_data = tf.math.divide(tf.math.subtract(Y_data, tf.math.reduce_min(Y_data)), tf.math.subtract(tf.math.reduce_max(Y_data), tf.math.reduce_min(Y_data)))

        return X_data, Y_data

train_data = tf.data.Dataset.from_generator(gen_file, output_types=(
            tf.float32, tf.uint8), output_shapes=(tf.TensorShape([512, 512, 1]), tf.TensorShape([512, 512, 1])))

train_data = train_data.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.batch(batch_size)
train_data = train_data.repeat(epochs)
train_data = iter(train_data)


val_data = tf.data.Dataset.from_generator(gen_file_val, output_types=(
            tf.float32, tf.uint8), output_shapes=(tf.TensorShape([512, 512, 1]), tf.TensorShape([512, 512, 1])))

val_data = val_data.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_data = val_data.batch(batch_size)
val_data = iter(val_data)

# train = preprocessing.DataGenerator(num_epochs=epochs, batch_size=batch_size).dataset
# val = preprocessing.DataGenerator(num_epochs=epochs, batch_size=batch_size).dataset


model = model.U_net((512, 512, 1), n_filters=16, dropout=None)
# model.summary()
# tf.keras.utils.plot_model(U_net(), to_file='model.png', show_shapes=True)

# Checkpoint for storing weights during learning
checkpoint = ModelCheckpoint(
    'weights/model_weights_{}e_{}bs'.format(epochs, batch_size), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-5, mode='min')


history = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_data, validation_steps=steps_per_epoch_val, verbose=1,
                   callbacks=[checkpoint])

# result = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.2,
#                    callbacks=[checkpoint, reduce_lr])


# plt.figure(figsize=(12.0, 5.0))
# plt.subplot(1, 2, 1, label='accuracy plot')
# plt.plot(result.history['accuracy'])
# plt.plot(result.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 1, label='loss plot')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2, label='dice loss plot')
plt.plot(history.history['dice_loss'])
plt.plot(history.history['val_dice_loss'])
plt.ylabel('diceloss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()