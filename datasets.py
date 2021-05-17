import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'TEST_HARD' }

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        try:
            image = image.convert('RGB')
        except:
            print('-->', self.images[i])
            image = Image.new('RGB', (image.size[0], image.size[1]))

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def zzzf_mean_and_std(loader):
    # https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/39?u=aditya_gupta
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for i, data in enumerate(loader):
        if (i % 100 == 0): print(i)
        data = data[0].squeeze(0)
        if (i == 0): size = data.size(1) * data.size(2)
        mean += data.sum((1, 2)) / size

    mean /= len(loader)
    print(mean)
    mean = mean.unsqueeze(1).unsqueeze(2)

    for i, data in enumerate(loader):
        if (i % 100 == 0): print(i)
        data = data[0].squeeze(0)
        std += ((data - mean) ** 2).sum((1, 2)) / size

    std /= len(loader)
    std = std.sqrt()
    print(std)
    return mean, std

def huud_mean_and_std(loader):
    # https: // discuss.pytorch.org / t / about - normalization - using - pre - trained - vgg16 - networks / 23560 / 19?u = aditya_gupta
    mean = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples

    temp = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        elementNum = data.size(0) * data.size(2) * data.size(3)
        data = data.permute(1,0,2,3).reshape(3, elementNum)
        temp += ((data - mean.repeat(elementNum,1).permute(1,0))**2).sum(1)/(elementNum*batch_samples)
        nb_samples += batch_samples

    std = torch.sqrt(temp/nb_samples)
    print(mean)
    print(std)
    return mean, std

def xwkuang_mean_and_std(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, boxes, labels, diff in loader:
        # if(cnt%100 == 0):
        #     print(cnt)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        # print(fst_moment, torch.sqrt(snd_moment - fst_moment ** 2), snd_moment)
        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)