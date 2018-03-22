import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
import numbers
import torch

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)

def image_train(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #ten crops for image when validation, input the data_transforms dictionary
  start_first = 0
  start_center = (resize_size - crop_size - 1) / 2
  start_last = resize_size - crop_size - 1
 
  return transforms.Compose([
    ResizeImage(resize_size),
    PlaceCrop(crop_size, start_center, start_center),
    transforms.ToTensor(),
    normalize
  ])

def image_test_10crop(resize_size=256, crop_size=224):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #ten crops for image when validation, input the data_transforms dictionary
  start_first = 0
  start_center = (resize_size - crop_size - 1) / 2
  start_last = resize_size - crop_size - 1
  data_transforms = {}
  data_transforms['val0'] = transforms.Compose([
      ResizeImage(resize_size),ForceFlip(),
      PlaceCrop(crop_size, start_first, start_first),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val1'] = transforms.Compose([
      ResizeImage(resize_size),ForceFlip(),
      PlaceCrop(crop_size, start_last, start_last),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val2'] = transforms.Compose([
      ResizeImage(resize_size),ForceFlip(),
      PlaceCrop(crop_size, start_last, start_first),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val3'] = transforms.Compose([
      ResizeImage(resize_size),ForceFlip(),
      PlaceCrop(crop_size, start_first, start_last),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val4'] = transforms.Compose([
      ResizeImage(resize_size),ForceFlip(),
      PlaceCrop(crop_size, start_center, start_center),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val5'] = transforms.Compose([
      ResizeImage(resize_size),
      PlaceCrop(crop_size, start_first, start_first),
      transforms.ToTensor(),
      normalize
  ])
  data_transforms['val6'] = transforms.Compose([
    ResizeImage(resize_size),
    PlaceCrop(crop_size, start_last, start_last),
    transforms.ToTensor(),
    normalize
  ])
  data_transforms['val7'] = transforms.Compose([
    ResizeImage(resize_size),
    PlaceCrop(crop_size, start_last, start_first),
    transforms.ToTensor(),
    normalize
  ])
  data_transforms['val8'] = transforms.Compose([
    ResizeImage(resize_size),
    PlaceCrop(crop_size, start_first, start_last),
    transforms.ToTensor(),
    normalize
  ])
  data_transforms['val9'] = transforms.Compose([
    ResizeImage(resize_size),
    PlaceCrop(crop_size, start_center, start_center),
    transforms.ToTensor(),
    normalize
  ])
  return data_transforms

