from .model import Model

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(Model):
    def __init__(self, input_size, hidden_size, output_size, n_layers, name):
        super().__init__(name=name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.init()

    def init_layer_stack(self):
        return None
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def init_criterion(self):
        return nn.CrossEntropyLoss()

    def init_optim(self):
        return torch.optim.SGD(params=self.parameters(), lr=0.01, momentum=0.0)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        output = self.linear(lstm_out)
        return output

    # def forward(self, x, lengths):
    #     print("input shape:", x.shape)
    #     packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

    #     lstm_out, (h_n, c_n) = self.lstm(packed_x)
    #     out = lstm_out[:, -1, :]
    #     out = self.linear(out)
    #     return out

    # def forward(self, x, lengths):
    #     # print("input shape:", x.shape)
        
    #     # Pack the padded sequence
    #     packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

    #     # Pass the packed input through the LSTM
    #     lstm_out, (h_n, c_n) = self.lstm(packed_x)
        
    #     # Unpack the output
    #     unpacked_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

    #     # Get the last output for each sequence in the batch
    #     # Here you can choose to use h_n or the last output directly
    #     out = unpacked_out[torch.arange(unpacked_out.size(0)), lengths - 1]  # This gets the last relevant output for each sequence

    #     # Alternatively, you can use h_n if you want the last hidden state
    #     # out = h_n[-1]  # Use the last hidden state of the last layer

    #     # Pass through the linear layer
    #     out = self.linear(out)

    #     return out