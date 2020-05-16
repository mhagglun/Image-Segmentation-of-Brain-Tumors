import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Unet import Unet, dice_score
from generator import DataGenerator
import util

test_data = DataGenerator('data/braintumorval/', epochs=1, batch_size=1)

N = 5   # Number of samples to plot


model = Unet(input_size=(512, 512, 1), n_filters=16, dropout=None)
# model.summary()
model.load_weights('model_weights_5e_1bs.h5')

predictions = model.predict(
    test_data.stream, steps=test_data.steps_per_epoch, verbose=1)


test_images = []
for idx, data in enumerate(test_data.stream):
    if idx == N:
        break
    test_images.append(data)

for idx, img in enumerate(test_images):

    true_mask = img[1]
    predicted_mask = predictions[idx]
    # Threshold predicted output
    predicted_mask[predicted_mask > 0.5] = 1
    predicted_mask[predicted_mask <= 0.5] = 0

    dice = dice_score(true_mask, predicted_mask)
    print(dice)

    true_mask = true_mask.numpy().reshape((512, 512))
    predicted_mask = predicted_mask.reshape((512, 512))
    util.plot_masks(img[0].numpy().reshape((512, 512)), true_mask, predicted_mask,
                    filename=None)
