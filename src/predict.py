import matplotlib.pyplot as plt
import numpy as np
import model
import preprocessing



dataset = preprocessing.Dataset().load_batch('data/images')

train_images = dataset['image'][0:2, :, :]
train_masks = dataset['mask']

train_images = np.expand_dims(train_images, axis=3)
train_masks = np.expand_dims(train_masks, axis=3)


model = model.U_net(input_size=(512, 512, 1), n_filters=32, dropout=None)
model.summary()
model.load_weights('weights')


predictions = model.predict(train_images, batch_size=1, verbose=1)

ground_truth_mask = dataset['mask'][0, :, :]
predicted_mask = predictions[0, :, :]

plt.imshow(predicted_mask.reshape(512, 512),
                    cmap='gray', interpolation='none')
plt.show()

preprocessing.display_image(train_images[0, :,:], predicted_mask)

preprocessing.display_image(train_images[0, :,:], ground_truth_mask)
