import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class VideoDataset(Dataset):
    def __init__(self, video_dir, label_map, transform=None, frame_rate=10):
        """
        Args:
            video_dir (str): Path to the directory containing videos.
            label_map (dict): Dictionary mapping class names to labels (e.g., {'fight': 0, 'normal': 1}).
            transform (callable, optional): Optional transform to be applied on a video.
            frame_rate (int): How many frames to sample per second.
        """
        self.video_dir = video_dir
        self.label_map = label_map
        self.transform = transform
        self.frame_rate = frame_rate
        self.video_files = []
        self.labels = []

        for label_name in os.listdir(video_dir):
            class_dir = os.path.join(video_dir, label_name)
            if os.path.isdir(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith((".mp4", ".avi")):
                        self.video_files.append(os.path.join(class_dir, video_file))
                        self.labels.append(self.label_map[label_name])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_rate == 0:
                frame = cv2.resize(frame, (112, 112))  # Resize frame for the CNN
                frames.append(frame)

            frame_count += 1

        cap.release()

        # Convert the list of frames to a NumPy array and then to a tensor
        if len(frames) > 0:
            frames = np.array(frames)  # Convert list of frames to numpy array
            frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Normalize and permute to (batch, channels, height, width)
        else:
            frames = torch.empty(0, 3, 112, 112)  # Handle empty frame case

        video_length = frames.size(0)

        if self.transform:
            frames = self.transform(frames)

        return frames, label, video_length  # Returning the length of the video


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    Args:
        batch: List of (frames, label, length) tuples.
    Returns:
        Padded frames, labels, and sequence lengths.
    """
    # Unpack the batch into frames, labels, and lengths
    frames, labels, lengths = zip(*batch)

    # Convert lists to tensors
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    # Sort by the length of the sequences in descending order (required by pack_padded_sequence)
    lengths, sorted_idx = lengths.sort(descending=True)
    frames = [frames[i] for i in sorted_idx]
    labels = labels[sorted_idx]

    # Pad the sequences so they all have the same length in the batch
    padded_frames = pad_sequence(frames, batch_first=True)

    return padded_frames, labels, lengths


# Example usage in a DataLoader
if __name__ == "__main__":
    video_dir = 'dataset/train'
    label_map = {'fight': 0, 'normal': 1}

    # Initialize the dataset
    dataset = VideoDataset(video_dir, label_map)

    # Initialize DataLoader with the custom collate_fn
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

    # Print out batch shape
    for batch in train_loader:
        frames, labels, lengths = batch
        print(f"Batch frames shape: {frames.shape}, Labels: {labels}, Lengths: {lengths}")
        break  # Print only the first batch for brevity
