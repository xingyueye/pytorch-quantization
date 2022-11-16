import os
import sys
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class ImageNetCalibDataset:
    """
    Creates batches of pre-processed images.
    """
    def __init__(self,
                 input,
                 calib_size,
                 interpolation='bilinear',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 cropt_pct=0.875):
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        normalize = transforms.Normalize(mean=mean,
                                         std=std)

        input_size = calib_size
        interp_mode = {
            'nearest': transforms.InterpolationMode.NEAREST,
            'bilinear': transforms.InterpolationMode.BILINEAR,
            'bicubic': transforms.InterpolationMode.BICUBIC,
            'box': transforms.InterpolationMode.BOX,
            'hamming': transforms.InterpolationMode.HAMMING,
            'lanczos': transforms.InterpolationMode.LANCZOS,
        }
        self.calib_transform = transforms.Compose([
            transforms.Resize(int(input_size / cropt_pct), interp_mode[interpolation]),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.convert(mode='RGB')
        image = self.calib_transform(image)
        return image

    def __getitem__(self, item):
        return self.preprocess_image(self.images[item])

    def __len__(self):
        return self.num_images

def get_val_loader(data_root,
                n_worker,
                batch_size,
                val_size=224,
                interpolation='bilinear',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                cropt_pct = 0.875
                ):
    '''
        split the train set into train / val for rl search
    '''
    print('=> Preparing val loader...')
    val_dir = os.path.join(data_root, 'val')

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    input_size = val_size
    interp_mode = {
        'nearest': transforms.InterpolationMode.NEAREST,
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'box': transforms.InterpolationMode.BOX,
        'hamming': transforms.InterpolationMode.HAMMING,
        'lanczos': transforms.InterpolationMode.LANCZOS,
    }
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size / cropt_pct), interp_mode[interpolation]),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])
    valset = datasets.ImageFolder(val_dir, test_transform)
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             num_workers=n_worker,
                                             pin_memory=True)


    return val_loader
