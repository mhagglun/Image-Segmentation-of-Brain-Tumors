import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import model
from preprocessing import DataGenerator


sns.set_style('darkgrid')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
#

# dataset = preprocessing.Dataset().load_batch('data/train_images')

batch_size = 1
epochs = 10

train_data = DataGenerator(
    'data/braintumor/', epochs=epochs, batch_size=batch_size)

val_data = DataGenerator('data/braintumorval/',
                         epochs=epochs, batch_size=batch_size)

model = model.U_net((512, 512, 1), n_filters=16, dropout=None)
# model.summary()
# tf.keras.utils.plot_model(U_net(), to_file='model.png', show_shapes=True)

# Checkpoint for storing weights during learning
checkpoint = ModelCheckpoint(
    'weights/model_weights_{}e_{}bs'.format(epochs, batch_size), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-5, mode='min')


history = model.fit_generator(train_data.stream, steps_per_epoch=train_data.steps_per_epoch, epochs=epochs, verbose=1,
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
