import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import numpy as np
import model
import preprocessing


dataset = preprocessing.Dataset().load_batch('data/images')

N = 10
train_images = dataset['image'][0:N, :, :]
train_masks = dataset['mask'][0:N, :, :]

train_images = np.expand_dims(train_images, axis=3)
train_masks = np.expand_dims(train_masks, axis=3)


model = model.U_net(input_size=(512, 512, 1), n_filters=32, dropout=None)
model.summary()
model.load_weights('weights')


predictions = model.predict(train_images, batch_size=1, verbose=1)

for n in range(N):
    true_mask = dataset['mask'][n, :, :].reshape(512, 512)
    predicted_mask = predictions[n, :, :].reshape(512, 512)

    predicted_mask = (predicted_mask - np.min(predicted_mask)) / \
        (np.max(predicted_mask) - np.min(predicted_mask))
    predicted_mask[predicted_mask > 0.5] = 1
    predicted_mask[predicted_mask <= 0.5] = 0

    preprocessing.display_image(train_images[n, :, :], predicted_mask)

    preprocessing.display_image(train_images[n, :, :], true_mask)


def dice_loss(y_true, y_pred):
    num = 2.0 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denom = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1.0 - (num + 1.0) / (denom + 1.0)

