import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

from generator import DataGenerator
from Unet import Unet
from Linknet import LinkNet
import util

sns.set_style('darkgrid')

BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT = None
N = 5   # Number of samples to plot


train_data = DataGenerator(
    'data/train', epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

val_data = DataGenerator('data/val',
                         epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

test_data = DataGenerator('data/test')

model = Unet((512, 512, 1), n_filters=16,
             learning_rate=LEARNING_RATE, dropout=DROPOUT).model

# model = LinkNet((512, 512, 1), num_classes=1,
#                 learning_rate=LEARNING_RATE).model

# model.summary()

# The directory to store the results in
folder = 'results/{}_{}e/'.format(model.name, EPOCHS)
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

tf.keras.utils.plot_model(
    model, to_file=folder+'{}.eps'.format(model.name), show_shapes=True)


# Checkpoint for storing weights during learning
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    folder+'model.h5'.format(model.name, EPOCHS, BATCH_SIZE), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5, mode='min')

# Stop learning when there has been no improvement in the tracked metric for 'patience' epochs
# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=1e-5, patience=5, mode='auto')

# Tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=folder+'logs', histogram_freq=1)

history = model.fit(train_data.stream, steps_per_epoch=train_data.steps_per_epoch, epochs=EPOCHS, validation_data=val_data.stream, validation_steps=val_data.steps_per_epoch, verbose=1,
                    callbacks=[checkpoint, reduce_lr, tensorboard_callback])

plt.figure(figsize=(20.0, 10.0))
plt.suptitle('{}'.format(model.name))
plt.subplot(1, 2, 1, label='loss plot')
plt.plot(np.arange(1, len(history.history['loss'])+1),history.history['loss'])
plt.plot(np.arange(1, len(history.history['val_loss'])+1), history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.subplot(1, 2, 2, label='dice score plot')
plt.plot(np.arange(1, len(history.history['dice_score'])+1), history.history['dice_score'])
plt.plot(np.arange(1, len(history.history['val_dice_score'])+1), history.history['val_dice_score'])
plt.ylabel('DICE score')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig(folder+'performance.pdf')
# plt.show()

predictions = model.predict(
    test_data.stream, steps=test_data.steps_per_epoch, verbose=1)


test_images, DICE = [], []

for data, pred in zip(test_data.stream, predictions):
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    dice = util.dice_score(data[1], pred)
    DICE.append(dice)
    test_images.append(data)


tf.print('DICE score on test set:', tf.reduce_mean(tf.stack(DICE)).numpy())
with open(folder+'dice_score.txt', 'w') as f:
    f.write('DICE score on test set: {:.3f}'.format(tf.reduce_mean(tf.stack(DICE)).numpy()))


# Plot images
for idx, img in enumerate(test_images,1):
    if idx > N:
        break

    true_mask = img[1]
    predicted_mask = predictions[idx-1]
    # Threshold predicted output
    predicted_mask[predicted_mask > 0.5] = 1
    predicted_mask[predicted_mask <= 0.5] = 0

    true_mask = true_mask.numpy().reshape((512, 512))
    predicted_mask = predicted_mask.reshape((512, 512))
    util.plot_masks(img[0].numpy().reshape((512, 512)), true_mask, predicted_mask,
                    filename=folder+'{}.pdf'.format(idx))