from .model import Model

import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        return hidden  # Return the last hidden state as the context for the decoder

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, y, hidden):
        embedded = self.embedding(y)  # Input to decoder
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden  # Output logits and hidden state

class Seq2Seq(Model):
    def __init__(self, name, encoder, decoder, max_inference_tokens=64):
        super().__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.max_inference_tokens = max_inference_tokens

    def init_layer_stack(self):
        # layer stack implementation is not necessary for this model
        # since it uses an overridden forward method
        return None

    def init_criterion(self):
        # ignore index zero, since padding is used for batching, and padded with zeros
        return nn.CrossEntropyLoss(ignore_index=0)

    def init_optim(self):
        # --- lr will be overridden ---
        # adam due to auto optimizing the momentum by tracking the exponential(weighted) moving averages
        # https://optimization.cbe.cornell.edu/index.php?title=Adam
        return Adam(params=self.parameters(), lr=0)
        
    def forward(self, x, x_lengths, y=None):
        hidden = self.encoder(x, x_lengths)  # Encode input
        outputs = []

        # Initialize the input to the decoder with the SOS token (0 could be used for EOS)
        input = torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device)  # Shape: [batch_size, 1]
        
        if y is not None:  # Training phase with teacher forcing
            for t in range(y.size(1)):  # Loop through the length of the output sequence
                output, hidden = self.decoder(input, hidden)
                outputs.append(output)

                # Use the actual next token from y as input for the next time step
                input = y[:, t].unsqueeze(1)  # Use the next token from y as input

                # Stop generating if EOS (0) is reached
                if (input == 0).all():
                    break
        
        else:  # Inference phase
            # for _ in range(self.max_inference_tokens):
            #     output, hidden = self.decoder(input, hidden)
            #     outputs.append(output)
            #     print(output)

            #     # Use the generated output as input for the next time step
            #     # Get the predicted token (using argmax, for example)
            #     input = output.argmax(dim=-1)  # Shape: [batch_size, 1]

            #     # Stop generating if EOS (0) is reached
            #     if (input == 0).all():
            #         break

            for _ in range(self.max_inference_tokens):
                output, hidden = self.decoder(input, hidden)
                outputs.append(output)

                # Use the generated output as input for the next time step
                input = output.argmax(dim=-1, keepdim=True)  # Ensure shape: [batch_size, 1]

                # Stop generating if EOS (0) is reached
                if (input == 0).all():
                    break
        
        # Concatenate outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, output_seq_len, output_size]
        return outputs