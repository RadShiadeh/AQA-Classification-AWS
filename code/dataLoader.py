import os
import torch
from torch.utils.data import Dataset
import pickle

class VideoDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, num_frames=16):
        self.root_dir = root_dir
        self.labels = self.load_labels(labels_file)
        self.video_ids = list(self.labels.keys())
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_path = os.path.join(self.root_dir, f"{video_id}.pkl")

        # Load frames from pickle file
        with open(video_path, 'rb') as file:
            frames_array = pickle.load(file)

        frames_tensor = torch.from_numpy(frames_array)

        # Ensure consistent number of frames
        if len(frames_tensor) < self.num_frames:
            frames_tensor = torch.cat([frames_tensor, torch.zeros(self.num_frames - len(frames_tensor), *frames_tensor.shape[1:])], dim=0)
        elif len(frames_tensor) > self.num_frames:
            frames_tensor = frames_tensor[:self.num_frames]

        classification_label, score_label = self.labels[video_id]

        classification_label_tensor = torch.tensor(classification_label, dtype=torch.float32)
        score_label_tensor = torch.tensor(score_label, dtype=torch.float32)

        return frames_tensor, classification_label_tensor, score_label_tensor

    def load_labels(self, labels_file):
        with open(labels_file, 'rb') as file:
            labels = pickle.load(file)
        return labels