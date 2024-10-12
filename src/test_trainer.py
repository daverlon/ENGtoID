from models.lstm import LSTMModel
from trainers.trainer import Trainer
from dataset.ENGtoID import ENGtoID

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    model = LSTMModel(1, 20, 5, 1, "lstm_test")
    trainer = Trainer(3, False)

    bs = 32

    dataset_train = ENGtoID(False, None)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=bs, shuffle=True, collate_fn=ENGtoID.collate_fn)

    trainer.fit(model, dataloader_train, None)

    # """
    for x, y, lengths in dataloader_train:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        print("in: x", x.shape, ", y:", y.shape)
        out = model(x, lengths)
        print("out:", out.shape)
        # exit()
    # """
