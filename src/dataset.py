from config import *

import torch
from torch.utils.data import Dataset as torchDataset

from datasets import Dataset as dsDataset

class ENGtoID(torchDataset):
    def __init__(self, valid=False, transform=None):

        self.transform = transform

        dataset_path = VALID_ENCODED_PATH if valid else TRAIN_ENCODED_PATH
        self.data = dsDataset.load_from_disk(dataset_path)["text"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        print(self.data[i]["eng"])
        x, y = torch.tensor(self.data[i]["eng"]), torch.tensor(self.data[i]["id"])

        if self.transform:
            x = self.transform(x)
        return x, y