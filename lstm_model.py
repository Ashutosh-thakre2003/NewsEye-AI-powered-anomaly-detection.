import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths=None):
        print(f"input shape : {x[0].shape}")
        """
        Args:
            x: Packed input sequence
            lengths: Original lengths of sequences before padding
        
        Returns:
            Final output of the model for classification
        """
        # Pass through the LSTM layer (input is packed)
        packed_out, (hn, cn) = self.lstm(x)

        # Optionally unpack the sequences if you need to use the entire output
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # Debugging statements
        print(f"LSTM output shape: {lstm_out.shape}")
        print(f"Hidden state shape: {hn.shape}")

        # Extract the output for the last valid time step (from the hidden state)
        # `hn[-1]` corresponds to the last layer’s hidden state for each sequence
        # `hn` has shape [num_layers * num_directions, batch_size, hidden_size]
        output = self.fc(hn[-1])

        return output

if __name__ == "__main__":
    # Test with dummy input
    batch_size = 2
    seq_len = 10
    input_size = 512
    hidden_size = 128
    output_size = 2
    num_layers = 1
    
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    print(model)
    
    # Dummy data
    x = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([seq_len] * batch_size)
    
    # Pack sequences
    packed_inputs = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    
    # Forward pass
    outputs = model(packed_inputs, lengths)
    print(f"Output shape: {outputs.shape}")
