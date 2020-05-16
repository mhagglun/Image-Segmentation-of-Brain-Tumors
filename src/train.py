from generator import DataGenerator
from Unet import Unet
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

sns.set_style('darkgrid')

batch_size = 1
epochs = 5

train_data = DataGenerator(
    'data/braintumor/', epochs=epochs, batch_size=batch_size)

val_data = DataGenerator('data/braintumorval/',
                         epochs=epochs, batch_size=batch_size)

model = Unet((512, 512, 1), n_filters=16, dropout=None)
# model.summary()
# tf.keras.utils.plot_model(U_net(), to_file='model.png', show_shapes=True)

# Checkpoint for storing weights during learning
checkpoint = ModelCheckpoint(
    'weights/model_weights_{}e_{}bs.h5'.format(epochs, batch_size), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5, mode='min')


history = model.fit(train_data.stream, steps_per_epoch=train_data.steps_per_epoch, epochs=epochs, validation_data=val_data.stream, validation_steps=val_data.steps_per_epoch, verbose=1,
                    callbacks=[checkpoint])

# plt.subplot(1, 2, 2, label='loss plot')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

# plt.subplot(1, 2, 1, label='dice loss plot')
# plt.plot(history.history['dice_score'])
# plt.plot(history.history['val_dice_score'])
# plt.ylabel('dice score')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
