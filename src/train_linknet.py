from Linknet import LinkNet

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# Training parameters
EPOCHS = 5
BATCH_SIZE = 8
ETA = 0.001 # Learning rate

IMAGE_SHAPE = (512, 512, 1)
