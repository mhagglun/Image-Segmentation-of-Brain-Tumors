import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams["axes.grid"] = False


def plot_masks(image_data, true_mask, predicted_mask, filename=None):
    """Plots the image and if specified, the mask and border. 
       Takes either a dictionary containing the image information or the corresponding ndarrays

    Arguments:
        image_data {ndarray} -- The image data to plot
        true_mask {ndarray} -- The ground truth tumor mask
        predicted_mask {ndarray} -- The predicted tumor mask

    Keyword Arguments:
        filename {str} -- If specified, the filepath for saving the image (default: {None})
    """
    # Configure colormap for overlay
    cmap = plt.cm.Reds
    red = cmap(np.arange(cmap.N))
    red[:, -1] = np.linspace(0, 1, cmap.N)
    red = ListedColormap(red)

    cmap = plt.cm.gray
    gray = ListedColormap(['white', 'gray', 'black'])
    norm = BoundaryNorm([1, 0, -1], cmap.N)

    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_data.reshape(512, 512),
               cmap='gray', interpolation='none')
    ax1.imshow(true_mask.reshape(512, 512), cmap=red,
               alpha=0.6, interpolation='none')
    ax1.set_title('True mask')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image_data.reshape(512, 512),
               cmap='gray', interpolation='none')
    ax2.imshow(predicted_mask.reshape(512, 512), cmap=red,
               alpha=0.6, interpolation='none')
    ax2.set_title('Predicted mask')

    ax3 = plt.subplot(1, 3, 3)
    mask = true_mask - predicted_mask
    ax3.imshow(mask, cmap='gray', interpolation='none')

    inv_mask = np.logical_not(np.logical_or(true_mask, predicted_mask))
    ax3.imshow(inv_mask, cmap=red, interpolation='none', alpha=1)
    patches = [mpatches.Patch(facecolor='white', label="FN", edgecolor='black'), mpatches.Patch(
        facecolor='gray', label="TP", edgecolor='black'), mpatches.Patch(facecolor='black', label="FP", edgecolor='black')]
    ax3.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)
    ax3.set_title('Difference between masks')

    # dice = dice_score(true_mask, predicted_mask)
    # fig.suptitle('DICE = {:.%2f}'.format(dice.numpy()))
    plt.tight_layout()

    if isinstance(filename, str):
        plt.savefig(filename)
    else:
        plt.show()


def dice_score(y_true, y_pred):

    y_pred = tf.math.greater(y_pred, 0.5)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    TP = tf.reduce_sum(tf.math.multiply(y_pred, y_true))
    diff = tf.math.subtract(y_pred, y_true)
    zero = tf.constant([0], dtype=tf.float32)
    FN = tf.math.less(diff, zero)
    FN = tf.reduce_sum(tf.cast(FN, tf.float32))
    FP = tf.math.greater(diff, zero)
    FP = tf.reduce_sum(tf.cast(FP, tf.float32))
    DICE = 2 * TP / (2 * TP + FN + FP)
    return DICE
