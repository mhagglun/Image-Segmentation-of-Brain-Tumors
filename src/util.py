import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns; sns.set()



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

    # cmap = plt.cm.RdBu
    cmap = plt.get_cmap('RdBu', 3)
    rgb = ListedColormap(['cornflowerblue', 'seagreen', 'firebrick'])
    norm = BoundaryNorm([1, 0, -1], cmap.N)

    cmap = plt.cm.gray
    white = cmap(np.arange(cmap.N))
    white[:, -1] = np.linspace(0, 1, cmap.N)
    white = ListedColormap(white)

    fig = plt.figure(figsize=(15.0, 6.0))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_data.reshape(512, 512),
               cmap='gray', interpolation='none')
    ax1.imshow(true_mask.reshape(512, 512), cmap=red,
               alpha=0.6, interpolation='none')

    ax1.axis('off')
    ax1.grid(b=None)
    ax1.set_title('True mask')


    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image_data.reshape(512, 512),
               cmap='gray', interpolation='none')
    ax2.imshow(predicted_mask.reshape(512, 512), cmap=red,
               alpha=0.6, interpolation='none')
    ax2.axis('off')
    ax2.grid(b=None)
    ax2.set_title('Predicted mask')


    ax3 = plt.subplot(1, 3, 3)
    mask = true_mask - predicted_mask
    dice = dice_score(true_mask, predicted_mask)
    ax3.imshow(mask, cmap=rgb, interpolation='none')

    inv_mask = np.logical_not(np.logical_or(true_mask, predicted_mask))
    ax3.imshow(inv_mask, cmap=white, interpolation='none', alpha=1)
    
    patches = [mpatches.Patch(facecolor='firebrick', label="FN", edgecolor='black'), mpatches.Patch(
        facecolor='seagreen', label="TP", edgecolor='black'), mpatches.Patch(facecolor='cornflowerblue', label="FP", edgecolor='black')]
    ax3.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)
    ax3.grid(color='lightgray')
    ax3.set_title('Difference between masks')
    ax3.set_xlabel('DICE score: {:.3f}'.format(tf.reduce_mean(tf.stack(dice)).numpy()), fontsize=12)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

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
