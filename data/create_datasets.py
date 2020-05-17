import os
import sys
import shutil
import random
from sklearn.model_selection import train_test_split


def create_datasets(directory='data/braintumor', split=(0.7, 0.2, 0.1), train_dir='data/train', val_dir='data/val', test_dir='data/test', symlink=False):
    files = os.listdir(directory)
    random.shuffle(files)
    folders = [train_dir, val_dir, test_dir]

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    train_samples = int(len(files) * split[0])
    val_samples = int(len(files) * split[1])

    train, remainder = files[:train_samples], files[train_samples:]
    val, test = remainder[:val_samples], remainder[val_samples:]

    for dataset, folder in zip([train, val, test], folders):
        for filename in dataset:
            filepath = os.path.join(directory, filename)
            if symlink:
                os.symlink(filepath, os.path.join(
                    folder, os.path.basename(filename)))
            else:
                shutil.copy(filepath, os.path.join(
                    folder, os.path.basename(filename)))


if __name__ == '__main__':
    # TODO: Allow user to pass arguments
    create_datasets()
