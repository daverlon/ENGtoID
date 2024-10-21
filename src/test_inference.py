from config import *

from models.seq2seq import Encoder, Decoder, Seq2Seq
from trainers.trainer import Trainer
from dataset.ENGtoID import ENGtoID

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    # setup dataset and dataloader
    dataset_valid = ENGtoID(True)
    dataloader_valid = DataLoader(dataset_valid, False)

    hidden_state_size = 128
    max_inference_tokens = 128
    encoder = Encoder(vocab_eng, hidden_state_size)
    decoder = Decoder(vocab_id, hidden_state_size)
    model = Seq2Seq("seq2seq_1", encoder, decoder, max_inference_tokens)
    model.init(0.0)
    model.load_model()
