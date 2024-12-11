import cv2
import os
import torch
from torchvision import transforms

def preprocess_videos(video_folder, output_folder, frame_rate=10, img_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_rate == 0:
                frame = cv2.resize(frame, img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count}.jpg")
                cv2.imwrite(save_path, frame)

            frame_count += 1
        cap.release()

def normalize_frames(frame_folder, output_folder):
    transform = transforms.Compose([transform.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    for frame_name in os.listdir(frame_folder):
        frame_path = os.path.join(frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        frame = transform(frame)
        torch.save(frame, os.path.join(output_folder, frame_name.split(".")[0] + ".pt"))

if __name__ == "__main__":
    preprocess_videos('dataset/videos', 'dataset/frames')
    normalize_frames('dataset/frames', 'dataset/normalized')
