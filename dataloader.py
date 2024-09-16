import pandas as pd
import numpy as np
from PIL import Image
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import random


class PairedTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img1, img2):
        seed = random.randint(0, 2 ** 32)
        torch.manual_seed(seed)
        img1 = self.base_transform(img1)

        torch.manual_seed(seed)
        img2 = self.base_transform(img2)

        return img1, img2


class Task1Dataset(Dataset):

    def __init__(self, transform, data_folder, csv_file):

        self.data_folder = data_folder
        self.transform = PairedTransform(transform)
        self.df = pd.read_csv(csv_file, sep=',')

        self.image_paths_t0 = self.df["image_at_ti"].tolist()
        self.image_paths_t1 = self.df["image_at_ti+1"].tolist()

        self.cases = self.df["case"].astype(int).tolist()
        self.label = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sample = {}

        path_t0 = os.path.join(self.data_folder, self.image_paths_t0[idx])
        path_t1 = os.path.join(self.data_folder, self.image_paths_t1[idx])

        img_t0 = Image.open(path_t0).convert("RGB")
        img_t1 = Image.open(path_t1).convert("RGB")

        if self.transform:
            img_t0, img_t1 = self.transform(img_t0, img_t1)

        sample['oct_slice_xt0'] = img_t0
        sample['oct_slice_xt1'] = img_t1
        sample['case_id'] = self.cases[idx]
        sample['label'] = self.label[idx]

        return sample


class Task2Dataset(Dataset):

    def __init__(self, transform, data_folder, csv_file):

        self.data_folder = data_folder
        self.transform = transform
        self.df = pd.read_csv(csv_file, sep=',')

        self.image_paths = self.df["image"].tolist()

        self.cases = self.df["case"].astype(int).tolist()
        self.label = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sample = {}

        path = os.path.join(self.data_folder, self.image_paths[idx])

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        sample['oct_slice_xt'] = img
        sample['case_id'] = self.cases[idx]
        sample['label'] = self.label[idx]

        return sample
