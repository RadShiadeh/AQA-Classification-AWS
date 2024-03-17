import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

class VideoDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, resize_shape=(256, 256), num_frames=16):
        self.root_dir = root_dir
        self.labels = self.load_labels(labels_file)
        self.video_ids = list(self.labels.keys())
        self.transform = transform
        self.resize_shape = resize_shape
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_path = os.path.join(self.root_dir, f"{video_id}.mp4")

        frames = self.read_video_frames(video_path)
        frames_resized = [transforms.functional.resize(Image.fromarray(frame), self.resize_shape) for frame in frames]

        frames_tensor = torch.stack([transforms.functional.to_tensor(frame) for frame in frames_resized])

        if len(frames_tensor) < self.num_frames:
            frames_tensor = torch.cat([frames_tensor, torch.zeros(self.num_frames - len(frames_tensor), 3, *self.resize_shape)], dim=0)
        elif len(frames_tensor) > self.num_frames:
            frames_tensor = frames_tensor[:self.num_frames]

        classification_label, score_label = self.labels[video_id]

        classification_label_tensor = torch.tensor(classification_label, dtype=torch.float32).view(1)
        score_label_tensor = torch.tensor(score_label, dtype=torch.float32).view(1)
        print(f"sample vid: [id, classification, score] = {[video_id, classification_label_tensor, score_label_tensor]}")

        return frames_tensor, classification_label_tensor, score_label_tensor

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as file:
            labels = json.load(file)
        return labels

    def read_video_frames(self, video_path):
        print("reading frames")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        print("got the frames and released the cap")
        return frames
