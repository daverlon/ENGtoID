from dataset.ENGtoID import ENGtoID

import torch
import torchvision

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == "__main__":

    dataset_train = ENGtoID(False, None)
    dataloader_train = DataLoader(dataset_train, 32, False, collate_fn=ENGtoID.collate_fn)
    dataset_valid = ENGtoID(True, False)
    dataloader_valid = DataLoader(dataset_valid, 32, False, collate_fn=ENGtoID.collate_fn)


    for batch_train in dataloader_train:
        break
    print(len(batch_train))
    x, y, xl, yl = batch_train
    print("------------ X ------------")
    print(x)
    print("------------ Y ------------")
    print(y)
    print("------------ XL ------------")
    print(xl)
    print("------------ YL ------------")
    print(yl)


    for batch_valid in dataloader_valid:
        break
    print(batch_valid)
    print(len(batch_valid))
    x, y, xl, yl = batch_valid
    print("------------ X ------------")
    print(x)
    print("------------ Y ------------")
    print(y)
    print("------------ XL ------------")
    print(xl)
    print("------------ YL ------------")
    print(yl)
