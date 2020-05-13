import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import numpy as np
import model
import preprocessing
import util

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# tf.debugging.set_log_device_placement(True)

dataset = preprocessing.Dataset().load_batch('data/test_images')


N = 10   # Number of samples to plot
indices = sorted(np.random.choice(len(dataset['image']), N, replace=False))

test_images = dataset['image'][indices, :, :]
test_masks = dataset['mask'][indices, :, :]

test_images = np.expand_dims(test_images, axis=3)
test_masks = np.expand_dims(test_masks, axis=3)

model = model.U_net(input_size=(512, 512, 1), n_filters=16, dropout=None)
model.summary()
model.load_weights('weights/model_weights_10e_8bs')

predictions = model.predict(test_images, batch_size=1, verbose=1)

for n in range(N):
    true_mask = test_masks[n].reshape(512, 512)
    predicted_mask = predictions[n].reshape(512, 512)

    # Threshold predicted output
    predicted_mask[predicted_mask > 0.5] = 1
    predicted_mask[predicted_mask <= 0.5] = 0

    util.plot_masks(test_images[n], true_mask, predicted_mask,
                    filename='results/10epochs_{}.png'.format(n+1))
