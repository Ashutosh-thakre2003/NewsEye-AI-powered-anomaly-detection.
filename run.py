import os
import cv2
import torch
from models.anomaly_detection import AnomalyDetectionModel
from nlp.nlp_generation import TextGenerator

def extract_frames(video_file, frame_rate=10):
    frames = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:  # Extract frame at every 'frame_rate' interval
            frame = cv2.resize(frame, (224, 224))  # Resize frame for the CNN input
            frames.append(frame)
        
        frame_count += 1
    cap.release()
    return frames

def run_system_on_dataset(dataset_dir, threshold=0.5):
    # Load trained model
    model = AnomalyDetectionModel()
    generator = TextGenerator()

    # Iterate over all videos in the dataset directory
    for video_file in os.listdir(dataset_dir):
        if video_file.endswith((".mp4", ".avi")):
            video_path = os.path.join(dataset_dir, video_file)
            print(f"Processing video: {video_file}")

            # Extract frames from the video
            frames = extract_frames(video_path)
            
            # Convert frames to tensors and normalize them before passing to the model
            frames_tensor = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0 for frame in frames]
            
            # Pass frames to the model and detect anomalies
            for frame_tensor in frames_tensor:
                anomaly_score = model.detect_anomaly([frame_tensor])  # Assuming detect_anomaly expects a list of tensors
                if anomaly_score > threshold:
                    article = generator.generate_news(f"Anomaly detected in {video_file}: Fighting")
                    print(article)
                    break  # Stop once an anomaly is detected in this video

if __name__ == "__main__":
    dataset_dir = 'path_to_your_dataset_directory'  # Replace with the path to your dataset folder
    run_system_on_dataset(dataset_dir)
