from config import *

from .model import Model

import torch
import torch.nn as nn
from torch.optim import Adam

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed_input)  # Get both hidden and cell states
        return hidden, cell  # Return hidden and cell states

class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, y, hidden, cell):
        embedded = self.embedding(y)  # Input to decoder
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # Pass both hidden and cell states
        output = self.fc(output)
        return output, hidden, cell  # Output logits and both hidden and cell states

class Seq2SeqLSTM(Model):
    def __init__(self, name, encoder, decoder, max_inference_tokens=64):
        super().__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.max_inference_tokens = max_inference_tokens

    def init_layer_stack(self):
        # Layer stack implementation is not necessary for this model
        return None

    def init_criterion(self):
        # Ignore index zero, since padding is used for batching, and padded with zeros
        return nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def init_optim(self):
        # --- lr will be overridden ---
        return Adam(params=self.parameters(), lr=0)
        
    def forward(self, x, x_lengths, y=None):
        hidden, cell = self.encoder(x, x_lengths)  # Encode input and get both hidden and cell states
        outputs = []

        # Initialize the input to the decoder with the SOS token (0 could be used for EOS)
        input = torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device)  # Shape: [batch_size, 1]
        
        if y is not None:  # Training phase with teacher forcing
            for t in range(y.size(1)):  # Loop through the length of the output sequence
                output, hidden, cell = self.decoder(input, hidden, cell)  # Pass both hidden and cell states
                outputs.append(output)

                # Use the actual next token from y as input for the next time step
                input = y[:, t].unsqueeze(1)  # Use the next token from y as input

                # Stop generating if EOS (0) is reached
                if (input == EOS_IDX).all():
                    break
        
        else:  # Inference phase
            for _ in range(self.max_inference_tokens):
                output, hidden, cell = self.decoder(input, hidden, cell)  # Pass both hidden and cell states
                outputs.append(output)

                input = output.argmax(dim=-1)  # Shape: [batch_size, 1]

                if (input == EOS_IDX).all():
                    break

        # Concatenate outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, output_seq_len, output_size]
        return outputs