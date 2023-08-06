
from typing import Dict, Any

from pathlib import Path

import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np


# This is the path to the folder containing the images
DEFAULT_IMAGE_FOLDER_PATH = Path('example_dataset/')

# This script is set up to crop first, then resize
CENTRE_CROP_SIZE = 200
RESIZED_IMAGE_SIZE = 100


# Set up the transforms on load of the data
DEFAULT_TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.CenterCrop(CENTRE_CROP_SIZE),
        transforms.Resize(RESIZED_IMAGE_SIZE),
        transforms.Grayscale(),  # Comment this out if you want to work with color images
        transforms.ToTensor()
    ]
)
# In this case, as we aren't doing any kind of random augmentation we 
# can use the same transforms for the test data as the train data
DEFAULT_TEST_TRANSFORMS = DEFAULT_TRAIN_TRANSFORMS 


def load_image_targets_from_csv(csv_path: Path, header: bool = True) -> Dict[str, Any]:
    """
    This function loads the image targets from a csv file. It assumes that the csv file
    has a header row and that the first column contains the image path and all the subsequent
    columns contain the target values which are bundled together into a numpy array.
    """
    image_targets = {}
    with csv_path.open('r') as f:
        lines = f.readlines()
        start_line = 0
        # If there is a header, skip the first line
        if header:
            header_line = lines[0].strip().split(',')
            print(f'Header line of csv {csv_path} : {header_line}')
            start_line = 1
        for line in lines[start_line:]:
            line = line.strip().split(',')
            image_path = line[0]
            image_targets[image_path] = np.array([float(x) for x in line[1:]], dtype=np.float32)
    return image_targets


class RegressionImageFolder(datasets.ImageFolder):
    """
    The regression image folder is a subclass of the ImageFolder class and is designed for 
    image regression tasks rather than image classification tasks. It takes in a dictionary
    that maps image paths to their target values.
    """
    def __init__(
        self, root: str, image_targets: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(root, *args, **kwargs)
        paths, _ = zip(*self.imgs)
        self.targets = [image_targets[path] for path in paths]
        self.samples = self.imgs = list(zip(paths, self.targets))


class RegressionTaskData:
    """
    This class is a wrapper for the data that is used in the regression task. It contains
    the train and test loaders.
    """
    def __init__(
        self,
        device,
        image_folder_path: Path = DEFAULT_IMAGE_FOLDER_PATH,
    ) -> None:
        self.image_folder_path = image_folder_path
        self.trainloader = self.make_trainloader()
        self.testloader = self.make_testloader()

    def make_trainloader(
            self, 
            train_transforms = DEFAULT_TRAIN_TRANSFORMS
        ) -> torch.utils.data.DataLoader:
        """
        Builds the train data loader
        """
        train_data = RegressionImageFolder(
            self.image_folder_path / 'train', 
            image_targets=load_image_targets_from_csv(self.image_folder_path / 'train.csv'),
            transform=train_transforms
        )
        # This constructs the dataloader that actually determins how images will be loaded in batches
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
        return trainloader


    def make_testloader(
            self, 
            test_transforms = DEFAULT_TRAIN_TRANSFORMS
        ) -> torch.utils.data.DataLoader:
        """
        Builds the test data loader
        """
        test_data = RegressionImageFolder(
            self.image_folder_path / 'test', 
            image_targets=load_image_targets_from_csv(self.image_folder_path / 'test.csv'),
            transform=test_transforms
        )
        # This constructs the dataloader that actually determins how images will be loaded in batches
        testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
        return testloader

    def visualise_image(self):
        """
        This function visualises a single image from the train set
        """
        images, targets = next(iter(self.trainloader))
        print(targets[0].shape)
        print(images[0].shape)
        plt.imshow(images[0][0, :, :])
        plt.show()


if __name__ == '__main__':
    data = RegressionTaskData()
    data.visualise_image()
