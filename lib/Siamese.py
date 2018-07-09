import torch.utils.data as data

from PIL import Image

import os
import os.path
import json
import cv2
import numpy as np


def crop_image(item, img ,length):
    bbox = [int(item['xmin']), int(item['ymin']), int(item['xmin'])+int(item['width']), int(item['ymin'])+int(item['height'])]
    h, w = img.shape[0:2]
    # img = img.astype(np.float32, copy=False)
    bbox[0] = max(0, bbox[0]-length)
    bbox[1] = max(0, bbox[1]-length)
    bbox[2] = min(w, bbox[2]+length)
    bbox[3] = min(h, bbox[3]+length)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def load_json(json_file):
    assert os.path.exists(json_file), \
            'json file not found at: {}'.format(json_file)
    with open(json_file, 'rb') as f:
        data = json.load(f)
    return data

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(pair_json, data_json, extensions):
    pairs = []
    for key in pair_json.keys():            
        a = data_json[key.strip().split(',')[0]]
        b = data_json[key.strip().split(',')[1]]
        scale_length = key.strip().split(',')[2]
        item = (a, b, pair_json[key], int(scale_length))
        pairs.append(item)

    return pairs


class DatasetSiamese(data.Dataset):

    def __init__(self, pair_json, data_json, image_json, loader, extensions, transform=None, target_transform=None):
        self.loader = loader
        pair_json = load_json(pair_json)
        data_json = load_json(data_json)
        image_json = load_json(image_json)

        self.image_container = {}
        self.__get_image(image_json)

        print len(pair_json)
        samples = make_dataset(pair_json, data_json, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))


        self.extensions = extensions

        self.samples = samples
        self.targets = [s[2] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        a, b, target, length = self.samples[index]
        sample_a, sample_b = self.image_container[str(a['img_id'])], self.image_container[str(b['img_id'])]
        # print sample_a

        sample_a = crop_image(a, sample_a, length)
        sample_b = crop_image(b, sample_b, length)
        # cv2.imwrite('/home/zhuyanjia/pytorch_examples/siamese/images/{}.jpg'.format(str(a['object_id'])+'_'+str(b['object_id'])+'_'+'0'+str(target)), sample_a)
        # cv2.imwrite('/home/zhuyanjia/pytorch_examples/siamese/images/{}.jpg'.format(str(a['object_id'])+'_'+str(b['object_id'])+'_'+'1'), sample_b)

        if self.transform is not None:
            sample_a = self.transform(sample_a)
            sample_b = self.transform(sample_b)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_a, sample_b, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __get_image(self, image_json):
        for i in range(len(image_json)):
            key = image_json.keys()[i]
            self.image_container[key] = self.loader(image_json[key])


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImageSiamese(DatasetSiamese):
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
    def __init__(self, pair_json, data_json, image_json, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageSiamese, self).__init__(pair_json, data_json, image_json, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
