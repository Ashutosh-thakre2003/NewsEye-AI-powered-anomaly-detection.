import torch
from models.anomaly_detection import AnomalyDetectionModel  # Your model file
from dataset_loader import VideoDataset, collate_fn  # Import your dataset loader and custom collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler  # Mixed precision tools

print("Starting...")

def train_model(epochs=10, batch_size=2, learning_rate=0.001):
    # Initialize the model
    model = AnomalyDetectionModel()
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Assuming binary classification (fight vs. normal)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the gradient scaler for mixed precision training
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Load the dataset and create DataLoader with the custom collate_fn
    video_dir = 'dataset/train'  # Path to your training dataset
    label_map = {'fight': 0, 'normal': 1}  # Map your labels to numbers
    train_dataset = VideoDataset(video_dir, label_map)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch_idx, (inputs, labels, lengths) in enumerate(train_loader):
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast() if scaler else torch.no_grad():
                # Pass the inputs through the CNN first (before packing)
                batch_size, sequence_length, channels, height, width = inputs.shape
                inputs = inputs.view(batch_size * sequence_length, channels, height, width)

                # Extract CNN features
                cnn_features = model.cnn(inputs)

                # Reshape CNN features back to (batch_size, sequence_length, feature_size)
                cnn_features = cnn_features.view(batch_size, sequence_length, -1)

                # Pack the CNN features using the provided lengths
                packed_inputs = pack_padded_sequence(cnn_features, lengths, batch_first=True, enforce_sorted=False)

                # Forward pass through the LSTM and final layers
                packed_outputs, _ = model.lstm(packed_inputs)

                # Unpack the output (if needed for loss calculation)
                lstm_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

                # Compute loss (use the last output for classification)
                loss = criterion(lstm_outputs[:, -1, :], labels)

            # Scale the loss for mixed precision training
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")

        # Print the loss after each epoch
        print(f"Epoch [{epoch + 1}/{epochs}] completed. Total Loss: {running_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    # Set the number of epochs, batch size, and learning rate
    train_model(epochs=20, batch_size=8, learning_rate=0.001)
