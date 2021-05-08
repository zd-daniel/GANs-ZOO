# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 0030 22:46
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : datasets.py
# @Software: PyCharm


import numpy as np
import os

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from utils import is_image_file


def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
              crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False):
    '''
    读取图像，是否做增强
    '''
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray is False and img.mode != 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode != 'L':
        img = img.convert('L')

    # 随机进行水平翻转
    if is_mirror and np.random.randint(0, 1) == 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    # 去掉左，上，右，下四个边上的行/列数
    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            #print([w,cropSize])
            cx1 = np.random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = np.random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w-crop_width)/2.))
            cy2 = cy1 = int(round((h-crop_height)/2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img


class MyDataset(Dataset):
    '''
    preprocess dataset
    '''
    def __init__(self, image_list, root_path,
                 input_height=128, input_width=None, output_height=128, output_width=None,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):
        super(MyDataset, self).__init__()

        self.root_path = root_path
        self.image_filenames = image_list
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.is_gray = is_gray

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.input_transform = transforms.Compose([
            transforms.ToTensor(), 
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        img = load_image(os.path.join(self.root_path, self.image_filenames[index]),
                         self.input_height, self.input_width, self.output_height, self.output_width,
                         self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)


class MyDataLoader(object):
    '''
    dataloader with next and iter
    '''
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.unlimit_gen = self.generator(True)

    def generator(self, inf=False):
        while True:
            data_loader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=self.drop_last)
            for images in data_loader:
                yield images
            if not inf:
                break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __iter__(self):
        return self.get_iter()

    def __len__(self):
        return len(self.dataset)//self.batch_size


def load_data(root_path, datatype='cifar10', batch_size=16, output_size=32):
    assert datatype in ['cifar10', 'mnist', 'celebAHQ'], '未知数据集'

    if datatype == 'celebAHQ':
        image_list = [x for x in os.listdir(root_path) if is_image_file(x)]
        train_list = image_list[:int(0.8 * len(image_list))]
        test_list = image_list[int(0.8 * len(image_list)):]
        assert len(train_list) > 0
        assert len(test_list) >= 0

        trainset = MyDataset(train_list, root_path, input_height=None, crop_height=None,
                             output_height=output_size, is_mirror=True)
        testset = MyDataset(test_list, root_path, input_height=None, crop_height=None,
                            output_height=output_size, is_mirror=False)
        trainloader = MyDataLoader(trainset, batch_size)
        testloader = MyDataLoader(testset, batch_size, shuffle=False, drop_last=False)
        classes = None

        return trainset, trainloader, testset, testloader, classes

    elif datatype == 'cifar10':
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'trunk')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        trainset = torchvision.datasets.CIFAR10(root=root_path, train=True,
                                                download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                               download=False, transform=transform)

    elif datatype == 'mnist':
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))
                                        ])
        trainset = torchvision.datasets.MNIST(root=root_path, train=True,
                                              download=False, transform=transform)
        testset = torchvision.datasets.MNIST(root=root_path, train=False,
                                             download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, pin_memory=True,
                             drop_last=True, num_workers=4)
    testloader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=4)

    return trainset, trainloader, testset, testloader, classes
