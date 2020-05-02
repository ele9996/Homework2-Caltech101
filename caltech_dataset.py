from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')
        self._class_finder(self.root, "BACKGROUND_Google")
        if not split.endswith(".txt"):
            split = split + ".txt"

        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), split)

        self._items = []
        with open(input_path, "r") as split_file:
            for file_line in split_file.readlines():
                line = file_line.replace("\n", "")
                if not line.startswith("BACKGROUND_Google"):
                    category, img = line.split("/")
                    self._items.append((pil_loader(os.path.join(self.root, category, img)),
                                        self.class_list.index(category)))

    def _class_finder(self, folder, folder_to_exclude):
        self.class_list = [d.name for d in os.scandir(folder) if d.is_dir() and d.name != folder_to_exclude]
        self.class_list.sort()

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self._items[index]  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self._items)# Provide a way to get the length (number of elements) of the dataset
        return length
