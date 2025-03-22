import os
import glob
import torchvision
import torch
from torch.utils import data as data
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CombinedDataset(data.Dataset):
    """
    Dataset object for simple paired LR/HR super-resolution training.

    Args:
        opt (dict): Contains:
            lr_path (str): Path to LR images.
            hr_path (str): Path to HR images.
            phase (str): 'train', 'val' or 'test'.
    """

    def __init__(self, opt):
        super(CombinedDataset, self).__init__()
        self.opt = opt
        self.lr_path = opt['sentinel2_path']  # This will actually point to your LR path
        self.hr_path = opt['naip_path']       # This will point to your HR path
        self.phase = opt.get('phase', 'train')

        if not (os.path.exists(self.lr_path) and os.path.exists(self.hr_path)):
            raise Exception("Please make sure the paths to LR and HR directories are correct.")

        self.lr_images = sorted(glob.glob(os.path.join(self.lr_path, '*.png')))
        self.hr_images = sorted(glob.glob(os.path.join(self.hr_path, '*.png')))

        assert len(self.lr_images) == len(self.hr_images), "Mismatch in number of LR and HR images."

        self.data_len = len(self.lr_images)
        print(f"Number of datapoints for split {self.phase}: {self.data_len}")

    def __getitem__(self, index):
        lr_img_path = self.lr_images[index]
        hr_img_path = self.hr_images[index]

        lr_img = torchvision.io.read_image(lr_img_path).float() / 255.0  # convert to float [0,1]
        hr_img = torchvision.io.read_image(hr_img_path).float() / 255.0

        return {'lr': lr_img, 'hr': hr_img, 'lq_path': lr_img_path, 'Index': index}

    def __len__(self):
        return self.data_len
