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
        x, y = torch.Tensor(self.data[i]["eng"]), torch.Tensor(self.data[i]["id"])

        if self.transform:
            x = self.transform(x)
        return x, y

    @staticmethod
    def collate_fn(batch):
        # print(batch)
        # x, y = zip(*batch)
        # padded_x = pad_sequence(x, batch_first=True, padding_value=0)
        # padded_y = pad_sequence(y, batch_first=True, padding_value=0)
        # lengths = [len(seq) for seq in x]
        # return padded_x, padded_y, torch.tensor(lengths)

        x, y = zip(*batch)

       # Pad input sequences to fixed_max_length
    padded_inputs = pad_sequence([torch.cat((seq, torch.tensor([0] * (fixed_max_length - len(seq))))) for seq in inputs], batch_first=True, padding_value=0)
    
    # Pad target sequences to fixed_max_length
    padded_targets = pad_sequence([torch.cat((seq, torch.tensor([0] * (fixed_max_length - len(seq))))) for seq in targets], batch_first=True, padding_value=0)

    return padded_inputs, padded_targets

