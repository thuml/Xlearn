#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path

class TextData():
  def __init__(self, text_file, label_file, source_batch_size=64, target_batch_size=64, val_batch_size=4):
    all_text = np.load(text_file)
    self.source_text = all_text[0:92664, :]
    self.target_text = all_text[92664:, :]
    self.val_text = all_text[0:92664, :]
    all_label = np.load(label_file)
    self.label_source = all_label[0:92664, :]
    self.label_target = all_label[92664:, :]
    self.label_val = all_label[0:92664, :]
    self.scaler = StandardScaler().fit(all_text)
    self.source_id = 0
    self.target_id = 0
    self.val_id = 0
    self.source_size = self.source_text.shape[0]
    self.target_size = self.target_text.shape[0]
    self.val_size = self.val_text.shape[0]
    self.source_batch_size = source_batch_size
    self.target_batch_size = target_batch_size
    self.val_batch_size = val_batch_size
    self.source_list = random.sample(range(self.source_size), self.source_size)
    self.target_list = random.sample(range(self.target_size), self.target_size)
    self.val_list = random.sample(range(self.val_size), self.val_size)
    self.feature_dim = self.source_text.shape[1]
    
  def next_batch(self, train=True):
    data = []
    label = []
    if train:
      remaining = self.source_size - self.source_id
      start = self.source_id
      if remaining <= self.source_batch_size:
        for i in self.source_list[start:]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
        self.source_list = random.sample(range(self.source_size), self.source_size)
        self.source_id = 0
        for i in self.source_list[0:(self.source_batch_size-remaining)]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
      else:
        for i in self.source_list[start:start+self.source_batch_size]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
      remaining = self.target_size - self.target_id
      start = self.target_id
      if remaining <= self.target_batch_size:
        for i in self.target_list[start:]:
          data.append(self.target_text[i, :])
          # no target label
          #label.append(self.label_target[i, :])
          self.target_id += 1
        self.target_list = random.sample(range(self.target_size), self.target_size)
        self.target_id = 0
        for i in self.target_list[0:self.target_batch_size-remaining]:
          data.append(self.target_text[i, :])
          #label.append(self.label_target[i, :])
          self.target_id += 1
      else:
        for i in self.target_list[start:start+self.target_batch_size]:
          data.append(self.target_text[i, :])
          #label.append(self.label_target[i, :])
          self.target_id += 1
    else:
      remaining = self.val_size - self.val_id
      start = self.val_id
      if remaining <= self.val_batch_size:
        for i in self.val_list[start:]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
        self.val_list = random.sample(range(self.val_size), self.val_size)
        self.val_id = 0
        for i in self.val_list[0:self.val_batch_size-remaining]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
      else:
        for i in self.val_list[start:start+self.val_batch_size]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
    data = self.scaler.transform(np.vstack(data))
    label = np.vstack(label)
    return torch.from_numpy(data).float(),torch.from_numpy(label).float()


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in xrange(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def ClassSamplingImageList(image_list, transform, return_keys=False):
    data = open(image_list).readlines()
    label_dict = {}
    for line in data:
        label_dict[int(line.split()[1])] = []
    for line in data:
        label_dict[int(line.split()[1])].append(line)
    all_image_list = {}
    for i in label_dict.keys():
        all_image_list[i] = ImageList(label_dict[i], transform=transform)
    if return_keys:
        return all_image_list, label_dict.keys()
    else:
        return all_image_list
