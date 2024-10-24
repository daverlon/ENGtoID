from config import *

from models.seq2seq_lstm import LSTMEncoder, LSTMDecoder, Seq2SeqLSTM
from dataset.ENGtoID import ENGtoID
from utils.coder import Coder
from utils.preprocessing import load_vocab_from_disk

import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu

if __name__ == "__main__":


    vocab_id = N_ID_TOKENS
    vocab_eng = N_ENG_TOKENS

    vocab = load_vocab_from_disk(VOCAB_PATH)
    coder_id = Coder(vocab["id"])
    coder_eng = Coder(vocab["eng"])

    # setup dataset and dataloader
    dataset_valid = ENGtoID(True)
    dataloader_valid = DataLoader(dataset_valid, 1, True)

    hidden_state_size = 512
    layers = 2
    max_inference_tokens = 128
    encoder = LSTMEncoder(vocab_id, hidden_state_size, layers)
    decoder = LSTMDecoder(vocab_id, hidden_state_size, layers)
    model = Seq2SeqLSTM("seq2seq_LSTM_1_extend", encoder, decoder, max_inference_tokens)
    model.init(0)
    model.load_model()

    loss_fn = nn.CrossEntropyLoss()

    total_score = 0

    for i, data in enumerate(tqdm(dataloader_valid)):

        x, y = data
        xl = torch.tensor([x.shape[1]])

        # print("-"*10,"Input","-"*10)
        # print(x)
        # print(coder_eng.decode(x.flatten().tolist()))
        # print("-"*10,"Target","-"*10)
        # print(y)
        # print(coder_id.decode(y.flatten().tolist()))

        model.eval()
        with torch.no_grad():
            out = model(x, xl, None).argmax(dim=-1)

            # print("-"*10,"Output","-"*10)
            out = out.flatten().tolist()
            out_decoded = coder_id.decode(out)
            # print(out_decoded)
            y_decoded = coder_id.decode(y.flatten().tolist())
            # print(y_decoded)
            score = sentence_bleu([y_decoded], out_decoded)
            total_score += score
            # print("SCORE:", score)
            # print()
            # print()

        # print()
        # print()

    print(total_score/len(dataloader_valid))

