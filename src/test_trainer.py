from config import *

from dataset.ENGtoID import ENGtoID
from models.seq2seq_lstm import LSTMEncoder, LSTMDecoder, Seq2SeqLSTM
from trainers.trainer import Trainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


hyper_params = {
    "bs": 48,
    "lr": 0.001,
    "epochs": 1
}

if __name__ == "__main__":

    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    # setup dataset and dataloader
    dataset_train = ENGtoID(False)
    dataset_valid = ENGtoID(True)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=hyper_params["bs"], shuffle=True, collate_fn=ENGtoID.collate_fn)

    # having a batch size for the valid dataloader makes the valid_epoch loop more similar to the epoch (train) loop
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=False)
    # dataloader_valid = DataLoader(dataset_valid, hyper_params["bs"], False, collate_fn=ENGtoID.collate_fn)

    hidden_state_size = 512
    layers = 1
    max_inference_tokens = 128
    encoder = LSTMEncoder(vocab_eng, hidden_state_size, num_layers=layers)
    decoder = LSTMDecoder(vocab_id, hidden_state_size, num_layers=layers)
    model = Seq2SeqLSTM("seq2seq_1", hyper_params, encoder, decoder, max_inference_tokens)
    model.init(hyper_params["lr"])
    model.load_model()


    # setup trainer
    trainer = Trainer(hyper_params["epochs"], True)

    trainer.fit(model, dataloader_train, dataloader_valid)