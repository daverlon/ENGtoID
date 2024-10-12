from dataset import ENGtoID

import torch
import torchvision

from torchvision.transforms import ToTensor

if __name__ == "__main__":

    dataset = ENGtoID(False, None)

    for i in range(20):
        print(dataset[i])