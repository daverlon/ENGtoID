from config import *

from .model import Model

import torch
import torch.nn as nn
from torch.optim import Adam

# https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)
        return hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, y, hidden, cell):
        embedded = self.embedding(y)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell

class Seq2SeqLSTM(Model):
    def __init__(self, name, hyper_params, encoder, decoder, max_inference_tokens=64):
        super().__init__(name=name, hyper_params=hyper_params)
        self.encoder = encoder
        self.decoder = decoder
        self.max_inference_tokens = max_inference_tokens

    def init_layer_stack(self):
        # no need for layer stack (custom forward pass)
        # forward pass uses input -> encoder -> decoder (recurrent decoder loop)
        return None

    def init_criterion(self):
        # ignore_index=PAD_IDX, ignore the padding token
        return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def init_optim(self):
        # lr will be overridden by self.init()
        return Adam(params=self.parameters(), lr=0)
        
    def forward(self, x, x_lengths, y=None):

        #
        # https://en.wikipedia.org/wiki/Long_short-term_memory
        # encode the input sequence using the trained encoder
        # this returns an LSTM hidden state and cell after the x sequence (ran on the entire sequence)
        #
        hidden, cell = self.encoder(x, x_lengths)
        #
        # the 'encoded' hidden state and cell are passed into the decoder
        # the decoder recurrently processes each output of itself
        # each time it is ran it leaves an output, this output is added to the output sequence container
        # this repeats until EOS or a limit is applied
        #

        # output sequence container
        outputs = []

        # init the input tensor
        # x.size: batch size,
        # dim1 = 1: single token holder
        input = torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device)
        
        #
        # https://en.wikipedia.org/wiki/Teacher_forcing
        # teacher forcing:
        # use the actual labels to generate predictions,
        # then back optimize from those predictions.
        #
        # instead of: completely random predictions with no guidance
        #
        if y is not None:
            #
            # t: index throughout 0-n
            # n: number of time-steps in y's second dimension
            # (y's first dimension is the batch size)
            # y is padded with PAD_IDX after EOS_IDX to match the same size for all samples
            #
            for t in range(y.size(1)):  # Loop through the length of the output sequence

                #
                # run the initial hidden state and cell (calculated from x)
                # into the decoder
                # the input is provided as zero initially
                #
                output, hidden, cell = self.decoder(input, hidden, cell)
                # the decoder generates an output as well as hidden state and cell tensors
                # add the hidden state to the output sequence container
                outputs.append(output)

                # replace the 'input' tensor with each timestep for each batch 
                #
                # y[:, t]
                # the ':' in the first dimension accesses each sequence in the batch
                # the 't' in the second dimension accesses each value at index t (this timestep)
                #
                input = y[:, t].unsqueeze(1)  # Use the next token from y as input

                # stop generating output tokens once an EOS token is generated
                if (input == EOS_IDX).all():
                    break
        
        #
        # if y is passed as None,
        # this forward loop is treated as inference
        # instead of using y to apply teacher forcing to the model's prediction
        # the model creates a 0-n token sequence
        # while generating the sequence, if EOS_IDX is chosen, the sequence returns
        #
        # the loop is mostly the same as above
        #
        else:
            for _ in range(self.max_inference_tokens):
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs.append(output)

                input = output.argmax(dim=-1)

                if (input == EOS_IDX).all():
                    break

        # concat each output token along the sequence dimension,
        # so tht outputs are shaped as:
        # [bs, seq_len, output_size]
        #
        # output_size: number of unique tokens in the output vocab (Indonesian)
        #
        return torch.cat(outputs, dim=1)