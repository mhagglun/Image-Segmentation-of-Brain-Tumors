import glob
import h5py
from skimage.transform import resize
import numpy as np

class Dataset:

    def __init__(self, directory='data/braintumor/'):
        """Constructor for the data set

        Keyword Arguments:
            directory {str} -- The path to the directory of the data files (default: {'data/braintumor/'})
        """
        super().__init__()
        self.directory = directory
        self.input_shape = (512, 512)

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

    def create_batch(self, num_samples=None, test_split=0.2, outputs=['data/train_images','data/test_images']):
        """Reads data and aggregates the data from .mat files in the specified directory

        Keyword Arguments:
            num_samples {int} -- The number of samples to aggregate or all in the directory if None (default: {None})
            output {str} -- Saves the aggregated data to the set file path if specified (default: {None})

        Returns:
            {dict} -- Returns the aggregated data as a dictionary
        """
        if num_samples is None:
            num_samples = len(glob.glob1(self.directory, '*.mat'))

        # images
        X = np.zeros((num_samples, self.input_shape[0], self.input_shape[1]))
        # tumor masks
        Y = np.zeros((num_samples, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((num_samples))

        for file_num, file in enumerate(glob.iglob(self.directory+'*.mat')):

            if file_num == num_samples:
                break

            image_data = self.read_file(file)

            if image_data['image'].shape != self.input_shape:
                image_data['image'] = resize(
                    image_data['image'], self.input_shape, mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)
                image_data['tumorMask'] = resize(
                    image_data['tumorMask'], self.input_shape, mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)

            X[file_num, :, :] = (image_data['image'] - np.min(image_data['image'])) / (
                np.max(image_data['image']) - np.min(image_data['image']))
            Y[file_num, :, :] = image_data['tumorMask']
            labels[file_num] = image_data['label']


        train_samples = int(num_samples * (1-test_split))
        rand = np.random.permutation(num_samples)   # shuffle indices of the data to select

        train_batch = {
            'image': X[rand[:train_samples], :, :],
            'mask': Y[rand[:train_samples], :, :],
            'label': labels[rand[:train_samples]],
        }

        test_batch = {
            'image': X[rand[train_samples:]],
            'mask': Y[rand[train_samples:]],
            'label': labels[rand[train_samples:]],
        }

        for output, batch in zip(outputs, [train_batch, test_batch]):
            hf = h5py.File(output, 'w')
            for key in batch:
                hf.create_dataset(key, data=batch[key])
            hf.close()

        return train_batch, test_batch

    def load_batch(self, filename):
        """Loads the data from the specified h5df file and returns its as a dictionary

        Arguments:
            filename {str} -- The path to the file

        Returns:
            [dict] -- A dictionary containing the loaded data
        """
        hf = h5py.File(filename, 'r')
        data = {key: hf[key] for key in hf.keys()}
        return data



# Example use
data = Dataset().create_batch(1250)
