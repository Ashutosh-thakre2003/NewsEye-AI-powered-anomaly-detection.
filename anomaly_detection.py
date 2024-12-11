import torch
import torch.nn as nn
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        self.cnn = CNNModel()  # Pre-trained CNN (e.g., ResNet50)
        self.lstm = LSTMModel(input_size=2048, hidden_size=128, output_size=2)  # Adjusted input_size
        self.fc = nn.Linear(128, 2)  # Final fully connected layer for classification

    def forward(self, frame_sequence, lengths):
        """
        Args:
            frame_sequence: Tensor of shape (batch_size, sequence_length, channels, height, width)
            lengths: List of sequence lengths (used for packing padded sequences)
        
        Returns:
            anomaly_score: Output class scores (batch_size, num_classes)
        """
        # Reshape frame_sequence to (batch_size * sequence_length, channels, height, width)
        batch_size, sequence_length, channels, height, width = frame_sequence.shape
        frame_sequence = frame_sequence.view(batch_size * sequence_length, channels, height, width)

        # Extract CNN features for each frame in the batch
        cnn_features = self.cnn(frame_sequence)  # (batch_size * sequence_length, feature_size)

        # Reshape CNN features back to (batch_size, sequence_length, feature_size)
        cnn_features = cnn_features.view(batch_size, sequence_length, -1)

        # Pack the padded sequences using the provided lengths
        packed_input = pack_padded_sequence(cnn_features, lengths, batch_first=True, enforce_sorted=False)

        # Pass through the LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Unpack the LSTM output
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # We use the last relevant LSTM output (based on the length of the sequence)
        # To do this, gather the last output from each sequence based on the lengths
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(batch_size, lstm_output.size(2))
        idx = idx.unsqueeze(1)  # Add dimension to match batch size
        lstm_last_outputs = lstm_output.gather(1, idx).squeeze(1)

        # Pass the last LSTM output through the fully connected layer for classification
        anomaly_score = self.fc(lstm_last_outputs)

        return anomaly_score

if __name__ == "__main__":
    # Example of how to use the model
    batch_size = 4
    sequence_length = 10
    channels, height, width = 3, 224, 224
    frame_sequence = torch.randn(batch_size, sequence_length, channels, height, width)
    
    # Sequence lengths for each sample in the batch
    lengths = [10, 8, 6, 5]  # Example sequence lengths, should be <= sequence_length
    
    model = AnomalyDetectionModel()
    print(model)

    # Pass the frame_sequence and lengths to the model
    output = model(frame_sequence, lengths)
    print(output.shape)  # Should output (batch_size, num_classes)
