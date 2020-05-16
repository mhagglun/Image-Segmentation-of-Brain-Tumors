import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from generator import DataGenerator
from Unet import Unet
from Linknet import LinkNet


sns.set_style('darkgrid')

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-3

train_data = DataGenerator(
    'data/train', epochs=EPOCHS, batch_size=BATCH_SIZE)

val_data = DataGenerator('data/val',
                         epochs=EPOCHS, batch_size=BATCH_SIZE)

# model = Unet((512, 512, 1), n_filters=16,
#              learning_rate=LEARNING_RATE, dropout=None).model

model = LinkNet((512, 512, 1), num_classes=1,
                learning_rate=LEARNING_RATE).model

# model.summary()
# tf.keras.utils.plot_model(
#     model.model, to_file='{}.png'.format(model.name), show_shapes=True)

# Checkpoint for storing weights during learning
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'weights/{}_weights_{}e_{}bs.h5'.format(model.name, EPOCHS, BATCH_SIZE), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5, mode='min')

# Stop learning when there has been no improvement in the tracked metric for 'patience' epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-4, patience=5, mode='auto')

# Tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='results/logs', histogram_freq=1)

history = model.fit(train_data.stream, steps_per_epoch=train_data.steps_per_epoch, epochs=EPOCHS, validation_data=val_data.stream, validation_steps=val_data.steps_per_epoch, verbose=1,
                    callbacks=[checkpoint, reduce_lr, early_stop, tensorboard_callback])

plt.suptitle('{}'.format(model.name))
plt.subplot(1, 2, 1, label='loss plot')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2, label='dice score plot')
plt.plot(history.history['dice_score'])
plt.plot(history.history['val_dice_score'])
plt.ylabel('DICE')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
