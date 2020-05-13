import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import model
import preprocessing

sns.set_style('darkgrid')

# tf.config.optimizer.set_jit(True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.debugging.set_log_device_placement(True)

dataset = preprocessing.Dataset().load_batch('data/train_images')

train_images = dataset['image']
train_masks = dataset['mask']
train_images = np.expand_dims(train_images, axis=3)
train_masks = np.expand_dims(train_masks, axis=3)

batch_size = 8
epochs = 10

model = model.U_net((512, 512, 1), n_filters=16, dropout=None)
# model.summary()
# tf.keras.utils.plot_model(U_net(), to_file='model.png', show_shapes=True)

# Checkpoint for storing weights during learning
checkpoint = ModelCheckpoint(
    'weights/model_weights_{}e_{}bs'.format(epochs, batch_size), save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=2, verbose=1, min_lr=1e-4, mode='min')

result = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.2,
                   callbacks=[checkpoint, reduce_lr])

# plt.figure(figsize=(12.0, 5.0))
# plt.subplot(1, 2, 1, label='accuracy plot')
# plt.plot(result.history['accuracy'])
# plt.plot(result.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')

# plt.subplot(1, 2, 2, label='loss plot')
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
