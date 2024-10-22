from config import *

import torch
from torch.utils.data import Dataset as torchDataset
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset as dsDataset

class ENGtoID(torchDataset):
    def __init__(self, valid=False, transform=None):

        self.transform = transform

        dataset_path = VALID_ENCODED_PATH if valid else TRAIN_ENCODED_PATH
        self.data = dsDataset.load_from_disk(dataset_path)["text"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print(self.data[i]["eng"], "-->", self.data[i]["id"])

        # tensor for int/long, Tensor for float
        x, y = torch.tensor(self.data[i]["eng"]), torch.tensor(self.data[i]["id"])

        if self.transform:
            x = self.transform(x)
        return x, y

    @staticmethod
    def collate_fn(batch):
        #print(batch)
        x, y = zip(*batch)
        padded_x = pad_sequence(x, batch_first=True, padding_value=PAD_IDX)
        padded_y = pad_sequence(y, batch_first=True, padding_value=PAD_IDX)
        lengths_x = [len(seq) for seq in x]
        lengths_y = [len(seq) for seq in y]
        return padded_x, padded_y, torch.tensor(lengths_x), torch.tensor(lengths_y)


