import glob
import h5py
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Dataset:
    def __init__(self):
        super().__init__()
        self.input_dim = 512 * 512
        self.num_labels = 3

    def read_file(self, filename):
        """Read the data from the .mat file and returns its contents as a dictionary

        Arguments:
            filename {str} -- The path to the .mat file to read

        Returns:
            [dict] -- A dictionary containing the data stored in the .mat file
        """
        data = {}
        with h5py.File(filename, 'r') as file:
            for struct in file:
                attributes = file[struct]
                for field in attributes:
                    if isinstance(field, str):
                        data[field] = attributes[field][()]
                    else:
                        data[struct] = field
        return data

    def create_batch(self, num_samples=None, filename=None):

        if num_samples is None:
            num_samples = len(glob.glob1('data/braintumor','*.mat'))

        X = np.zeros((self.input_dim, num_samples))        # Images
        Y = np.zeros((self.input_dim, num_samples))        # tumorMasks
        labels = np.zeros((num_samples))

        for file_num, file in enumerate(glob.iglob("data/braintumor/*.mat")):

            if file_num == num_samples: 
                break

            image_data = self.read_file(file)

            if image_data['image'].shape != (512, 512):
                image_data['image'] = resize(image_data['image'], (512,512))
                image_data['tumorMask'] = resize(image_data['tumorMask'], (512,512))

            X[:, file_num] = image_data['image'].flatten()
            Y[:, file_num] = image_data['tumorMask'].flatten()
            labels[file_num] = image_data['label']

        batch = {
            'images': X,
            'masks': Y,
            'labels': labels,
        }

        if filename is not None:

            hf = h5py.File(filename, 'w')
            for key in batch:
                hf.create_dataset(key, data=batch[key])
            hf.close()

        return batch


    def load_batch(self, filename):
        hf = h5py.File(filename, 'r')

        data = {key: hf[key] for key in hf.keys()}
        return data




def display_image(image_data, mask=None, show_mask=True, show_border=True):

    # Configure colormap for overlay
    cmap = plt.cm.Reds
    red = cmap(np.arange(cmap.N))
    red[:, -1] = np.linspace(0, 1, cmap.N)
    red = ListedColormap(red)

    if isinstance(image_data, dict):
        plt.imshow(image_data['image'], cmap='gray')
        if show_mask:
            plt.imshow(image_data['tumorMask'], cmap=red, alpha=0.5)

        if show_border:
            # TODO: Convert tumorBorder data [x1, y1, x2, y2 ...] into plottable image/outline
            # plt.imshow(image_data['tumorBorder'], cmap='jet', alpha=0.5)
            pass

    else:
        plt.imshow(image_data.reshape(512, 512),
                   cmap='gray', interpolation='none')
        if mask is not None:

            image_mask = np.ma.masked_where(
                mask.reshape(512, 512) > 0, mask.reshape(512, 512))
            plt.imshow(mask.reshape(512, 512), cmap=red,
                       alpha=0.6, interpolation='none')

    plt.show()


# Example use

# data1 = Dataset().create_batch(1000, filename='data/images_compressed')
# display_image(data1['images'][:, 0], data1['masks'][:, 0])

# data = Dataset().load_batch('data/images_compressed')
# display_image(data['images'][:, 950])

# img = Dataset().read_file('data/braintumor/955.mat')
# display_image(img)
