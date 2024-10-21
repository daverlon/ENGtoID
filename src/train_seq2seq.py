from config import *

from dataset.ENGtoID import ENGtoID
from models.seq2seq import Encoder, Decoder, Seq2Seq
from trainers.trainer import Trainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    hyper_params = {
        "bs": 64,
        "lr": 0.001,
        "epochs": 5
    }

    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    # setup dataset and dataloader
    dataset_train = ENGtoID(False)
    dataset_valid = ENGtoID(True)
    dataloader_train = DataLoader(dataset_train, hyper_params["bs"], True, collate_fn=ENGtoID.collate_fn)
    dataloader_valid = DataLoader(dataset_valid, hyper_params["bs"], False, collate_fn=ENGtoID.collate_fn)

    # setup model
    hidden_state_size = 128
    max_inference_tokens = 128
    encoder = Encoder(vocab_eng, hidden_state_size)
    decoder = Decoder(vocab_id, hidden_state_size)
    model = Seq2Seq("seq2seq_1", encoder, decoder, max_inference_tokens)
    model.init()

    # setup trainer
    trainer = Trainer(5, True)

    trainer.fit(model, dataloader_train, dataloader_valid)