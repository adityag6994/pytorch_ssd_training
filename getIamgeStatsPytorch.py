import torch.utils.data
from datasets import *
from utils import *

# Data parameters
data_folder = 'data/rafeeq/'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
batch_size = 1  # batch size
workers = 1  # number of workers for loading data in the DataLoader

# Custom dataloaders
train_dataset = PascalVOCDataset(data_folder,
                                 split='test',
                                 keep_difficult=keep_difficult)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=train_dataset.collate_fn, num_workers=workers,
                                           pin_memory=True)  # note that we're passing the collate function here

# find mean variance of dataset
mean, std = zzzf_mean_and_std(train_loader)
# mean, std = xwkuang_mean_and_std(train_loader)
print(mean, std)

## test hard | zzzf || xwkuang
# tensor([0.5881, 0.5617, 0.4820])
# tensor([0.2968, 0.3004, 0.2938])

## train
# mean = [0.4898, 0.4867, 0.4050]
# std = [0.2774, 0.2832, 0.2501]