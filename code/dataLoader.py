import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

class VideoDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, resize_shape=(480, 480), num_frames=100):
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

        # Read video frames
        frames = self.read_video_frames(video_path)

        # Resize frames to a consistent size
        frames_resized = [transforms.functional.resize(Image.fromarray(frame), self.resize_shape) for frame in frames]

        # Check if frames_resized is empty
        if not frames_resized:
            return None

        # Pad or trim frames to have the same size (num_frames)
        frames_resized = self.pad_or_trim_frames(frames_resized)

        # Convert resized frames to a list of tensors (if not already tensors)
        frames_resized = [transforms.functional.to_tensor(frame) if not isinstance(frame, torch.Tensor) else frame for frame in frames_resized]

        # Convert frames_resized to a torch tensor
        frames = torch.stack(frames_resized)

        # Get classification and score from labels
        classification, score = self.labels[video_id]

        # Apply transformations if provided
        if self.transform:
            frames = self.transform(frames)

        # Convert frames and labels to torch tensors
        classification = torch.tensor(classification, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)

        # Create data dictionary
        data = {
            'video': frames,
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



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
# labels_path = "../../dissData/labels/all_labels.json"
# sample_vids = "../../dissData/allVids"

# video_dataset = VideoDataset(sample_vids, labels_path, transform=None)

# Accessing a specific sample
# sample_index = 0
# sample_frames, sample_label = video_dataset[sample_index]

# # Print the shapes for demonstration
# batch_size = 4
# data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# # Iterate through the DataLoader
# for batch_frames, batch_labels in data_loader:
#     # Print information about the batch
#     print(f"Batch Frames Shape: {batch_frames.shape}")
#     print(f"Batch Labels: {batch_labels}")

#     # Access individual samples within the batch
#     for frames, label in zip(batch_frames, batch_labels):
#         print(f"Sample Frames Shape: {frames.shape}")
#         print(f"Sample Label: {label}")








# Class Definition:

# VideoDataset is a subclass of DataLoader, but it's more appropriate to inherit from torch.utils.data.Dataset.
# It is designed to load video data for a PyTorch model.
# Initialization:

# The __init__ method initializes the dataset with necessary parameters such as the data directory (data_dir), JSON file with video paths and labels (json_file), optional transformation (transform), target size for resizing frames (target_size), and the desired number of frames (num_frames).
# Length Method:

# The __len__ method returns the number of video samples in the dataset.
# Video Loading and Processing Methods:

# resize_frame: Resizes a frame to the specified target size.
# pad_or_trim_frames: Ensures that the video has the desired number of frames by padding or trimming.
# load_video_frames: Loads frames from a video file using OpenCV.
# __getitem__ Method:

# The __getitem__ method loads and processes a single video sample specified by the given index.
# It loads video frames, applies transformations, resizes frames, and ensures the correct number of frames.
# Returns a tuple containing the processed frames and the corresponding label.
# Usage Example:

# The script demonstrates how to create an instance of VideoDataset and access individual samples using a PyTorch DataLoader.
# The commented-out code at the end provides an example of using the dataset with a DataLoader and iterating through batches.