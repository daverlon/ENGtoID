from config import *

from dataset.ENGtoID import ENGtoID
from models.seq2seq_rnn import RNNEncoder, RNNDecoder, Seq2SeqRNN
from trainers.trainer import Trainer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    hyper_params = {
        "bs": 32,
        "lr": 0.001,
        "epochs": 5
    }

    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    # setup dataset and dataloader
    dataset_train = ENGtoID(False)
    # dataset_valid = ENGtoID(True)
    dataloader_train = DataLoader(dataset_train, hyper_params["bs"], True, collate_fn=ENGtoID.collate_fn)

    # setup model
    hidden_state_size = 256
    layers = 1
    max_inference_tokens = 128
    # encoder = Encoder(vocab_eng, hidden_state_size, num_layers=layers)
    # decoder = Decoder(vocab_id, hidden_state_size, num_layers=layers)
    encoder = RNNEncoder(vocab_id, hidden_state_size, layers)
    decoder = RNNDecoder(vocab_id, hidden_state_size, layers)
    model = Seq2SeqRNN("seq2seq_RNN_2", encoder, decoder, max_inference_tokens)
    model.init(hyper_params["lr"])

    # setup trainer
    trainer = Trainer(hyper_params["epochs"], True)

    trainer.fit(model, dataloader_train, None)