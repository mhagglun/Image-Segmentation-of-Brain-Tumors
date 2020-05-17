import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from Unet import Unet, dice_score
from Linknet import LinkNet
from generator import DataGenerator
import util

BATCH_SIZE = 1
EPOCHS = 30
N = 5   # Number of samples to plot

test_data = DataGenerator('data/test')

model = Unet((512, 512, 1), n_filters=64, dropout=None).model
# model = LinkNet((512, 512, 1), nf=64, num_classes=1).model

# model.summary()
model.load_weights(
    'results/{}_{}e/model.h5'.format(model.name, EPOCHS, BATCH_SIZE))

predictions = model.predict(
    test_data.stream, steps=test_data.steps_per_epoch, verbose=1)


test_images, DICE = [], []

for data, pred in zip(test_data.stream, predictions):
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    dice = dice_score(data[1], pred)
    DICE.append(dice)
    test_images.append(data)

tf.print('DICE score on test set:', np.mean(DICE))


# Plot images
for idx, img in enumerate(test_images, 1):
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
                    filename='results/{}_{}e/{}.pdf'.format(model.name, EPOCHS, idx))
    plt.show()
