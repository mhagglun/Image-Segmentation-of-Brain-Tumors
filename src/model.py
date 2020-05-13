from keras.models import Model
from keras.optimizers import *
from keras.layers import *


def U_net(input_size=(512, 512, 1), n_filters=16, dropout=None):

    useDropout = False
    if isinstance(dropout, (float, int)) and (dropout <= 1.0 and dropout >= 0.0):
        useDropout = True

    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(inputs)
    if useDropout:
        c1 = Dropout(dropout)(c1)
    c1 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(p1)
    if useDropout:
        c2 = Dropout(dropout)(c2)
    c2 = Conv2D(n_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(n_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(p2)
    if useDropout:
        c3 = Dropout(dropout)(c3)
    c3 = Conv2D(n_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(n_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(p3)
    if useDropout:
        c4 = Dropout(dropout)(c4)
    c4 = Conv2D(n_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(n_filters*16, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(p4)
    if useDropout:
        c5 = Dropout(dropout)(c5)
    c5 = Conv2D(n_filters*16, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(n_filters*8, (2, 2),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(n_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u6)
    if useDropout:
        c6 = Dropout(dropout)(c6)
    c6 = Conv2D(n_filters*8, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c6)

    u7 = Conv2DTranspose(n_filters*4, (2, 2),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(n_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u7)
    if useDropout:
        c7 = Dropout(dropout)(c7)
    c7 = Conv2D(n_filters*4, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c7)

    u8 = Conv2DTranspose(n_filters*2, (2, 2),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(n_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u8)
    if useDropout:
        c8 = Dropout(dropout)(c8)
    c8 = Conv2D(n_filters*2, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c8)

    u9 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(u9)
    if useDropout:
        c9 = Dropout(dropout)(c9)
    c9 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',
                padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(lr=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


def dice_loss(y_true, y_pred):
    num = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denom = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - num / denom
