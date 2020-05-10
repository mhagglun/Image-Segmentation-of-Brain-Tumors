from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import model
import preprocessing

dataset = preprocessing.Dataset().load_batch('data/images')

train_images = dataset['image']
train_masks = dataset['mask']

train_images = np.expand_dims(train_images, axis=3)
train_masks = np.expand_dims(train_masks, axis=3)


batch_size = 1
epochs = 1

model = model.U_net((512, 512, 1), n_filters=32, dropout=None)
model.summary()

# Checkpoint for storing weights during learning
checkpoint = ModelCheckpoint(
    'weights', save_best_only=True, monitor='val_loss', mode='min')

# Reduces learning rate when a metric has stopped improving
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=1e-4, mode='min')

result = model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=0.1,
                   callbacks=[checkpoint, reduce_lr_loss])
