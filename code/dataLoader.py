import os
import json
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import torch
import cv2

class VideoDataset(DataLoader):
    def __init__(self, data_dir, json_file, transform=None, target_size=(480, 480), num_frames = 300):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.num_frames = num_frames

        # Load video paths and labels from JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

        self.video_ids = list(data.keys())
        self.labels = list(data.values())

    def __len__(self):
        return len(self.video_ids)
    
    def resize_frame(self, frame):
        return F.resize(frame, self.target_size)
    
    def pad_or_trim_frames(self, frames):
        num_frames = frames.shape[0]

        if num_frames < self.num_frames:
            padding = torch.zeros(self.num_frames - num_frames, *frames.shape[1:], dtype=frames.dtype)
            frames = torch.cat([frames, padding])
        elif num_frames > self.num_frames:
            frames = frames[:self.num_frames]
        
        return frames
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.target_size:
                frame = cv2.resize(frame, self.target_size[::-1])
            
            frame = F.to_tensor(frame)
            frames.append(frame)
            
        cap.release()
        frames = torch.stack(frames)
        return frames

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        label = self.labels[index]

        # Load video frames using OpenCV
        video_path = os.path.join(self.data_dir, f'{video_id}.mp4')
        frames = self.load_video_frames(video_path)

        # Apply transformations if provided
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        frames = [self.resize_frame(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = self.pad_or_trim_frames(frames)

        return frames, label

labels_path = "../../dissData/labels/all_labels.json"
sample_vids = "../../dissData/allVids"

video_dataset = VideoDataset(sample_vids, labels_path, transform=None)

# Accessing a specific sample
sample_index = 0
sample_frames, sample_label = video_dataset[sample_index]

# Print the shapes for demonstration
batch_size = 4
data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch_frames, batch_labels in data_loader:
    # Print information about the batch
    print(f"Batch Frames Shape: {batch_frames.shape}")
    print(f"Batch Labels: {batch_labels}")

    # Access individual samples within the batch
    for frames, label in zip(batch_frames, batch_labels):
        print(f"Sample Frames Shape: {frames.shape}")
        print(f"Sample Label: {label}")