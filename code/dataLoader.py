import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

class VideoDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, resize_shape=(256, 256), num_frames=16, overlap=1):
        self.root_dir = root_dir
        self.labels = self.load_labels(labels_file)
        self.video_ids = list(self.labels.keys())
        self.transform = transform
        self.resize_shape = resize_shape
        self.num_frames = num_frames
        self.overlap = overlap

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_path = os.path.join(self.root_dir, f"{video_id}.mp4")

        # Read video frames
        frames = self.read_video_frames(video_path)

        # Resize frames to a consistent size
        frames_resized = [transforms.functional.resize(Image.fromarray(frame), self.resize_shape) for frame in frames]

        # Check if frames_resized is empty
        if not frames_resized:
            # Skip this sample and move to the next one
            return self.__getitem__((idx + 1) % len(self))

        # Pad or trim frames to have the same size (num_frames)
        frames_resized = self.pad_or_trim_frames(frames_resized)

        # Extract overlapping segments
        video_clip_segments = []
        for i in range(0, len(frames_resized) - 16 + self.overlap, self.overlap):
            clip_segment = frames_resized[i:i + 16]
            video_clip_segments.append(clip_segment)

        # Convert each clip segment to a torch tensor
        video_clip_segments = [torch.stack([transforms.functional.to_tensor(frame) for frame in clip_segment]) for clip_segment in video_clip_segments]

        # Convert video_clip_segments to a torch tensor
        video_clip = torch.stack(video_clip_segments)

        # Get classification and score from labels
        classification, score = self.labels[video_id]

        # Apply transformations if provided
        if self.transform:
            video_clip = self.transform(video_clip)

        # Convert frames and labels to torch tensors
        classification = torch.tensor(classification, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)

        # Create data dictionary
        data = {
            'video': video_clip,
            'classification': classification,
            'score': score
        }

        return data


    def load_labels(self, labels_file):
        with open(labels_file, 'r') as file:
            labels = json.load(file)
        return labels

    def read_video_frames(self, video_path):
        # Use OpenCV to read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def pad_or_trim_frames(self, frames):
        # Check if frames list is empty
        if not frames:
            return frames

        # Convert the first frame to a tensor to get the shape
        first_frame_tensor = transforms.functional.to_tensor(frames[0])

        # Trim or pad frames to have the same size (num_frames)
        if len(frames) < self.num_frames:
            # Padding frames with dark frames
            frames += [torch.zeros_like(first_frame_tensor) for _ in range(self.num_frames - len(frames))]
        else:
            # Trimming frames
            frames = frames[:self.num_frames]

        return frames