from config import *

from models.seq2seq_rnn import Encoder, Decoder, Seq2Seq
from trainers.trainer import Trainer
from dataset.ENGtoID import ENGtoID
from preprocessing.coder import Coder
from preprocessing.preprocessing import load_vocab_from_disk

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":

    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    vocab = load_vocab_from_disk(VOCAB_PATH)
    coder_id = Coder(vocab["id"])
    coder_eng = Coder(vocab["eng"])

    # setup dataset and dataloader
    dataset_valid = ENGtoID(True)
    dataloader_valid = DataLoader(dataset_valid, 1, True)

    hidden_state_size = 128
    max_inference_tokens = 128
    encoder = Encoder(vocab_eng, hidden_state_size)
    decoder = Decoder(vocab_id, hidden_state_size)
    model = Seq2Seq("seq2seq_1", encoder, decoder, max_inference_tokens)
    model.init(0.0)
    model.load_model()

    loss_fn = nn.CrossEntropyLoss()

    for i, data in enumerate(dataloader_valid):

        x, y = data
        xl = torch.tensor([x.shape[1]])

        print("-"*10,"Input","-"*10)
        print(x)
        print(coder_eng.decode(x.flatten().tolist()))
        print("-"*10,"Target","-"*10)
        print(y)
        print(coder_id.decode(y.flatten().tolist()))

        model.eval()
        with torch.no_grad():
            out = model(x, xl, None).argmax(dim=-1)
            out = out.flatten().tolist()
            print("-"*10,"Output","-"*10)
            print(out)
            print(coder_id.decode(out))

        print()
        print()

        if i > 10: break
