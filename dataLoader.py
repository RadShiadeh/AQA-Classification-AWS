import os
import json
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        with open(json_file, 'r') as file:
            data = json.load(file)

        self.video_ids = list(data.keys())
        self.labels = list(data.values())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        label = self.labels[index]

        # Load video frames using OpenCV
        video_path = os.path.join(self.data_dir, f'{video_id}.mp4')
        video, audio, info = read_video(video_path, pts_unit="sec")
        
        # Use only the video frames for simplicity (you might use audio too)
        frames = video.permute(0, 3, 1, 2)

        # Apply transformations if provided
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return frames, label

# Example usage:
data_dir = 'allVids' #replace with video samples path
json_file = 'all_labels.json' #replace with labels path
transform = None  

video_dataset = VideoDataset(data_dir, json_file, transform=transform)

# Accessing a specific sample
sample_index = 0
sample_frames, sample_label = video_dataset[sample_index]

# Print the shapes for demonstration
print(f"Video Frames Shape: {sample_frames.shape}")
print(f"Label: {sample_label}")