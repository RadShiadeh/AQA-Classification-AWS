from dataLoader import VideoDataset


labels_path = "../labels/complete_labels.json"
sample_vids = "../../dissData/allVids"

# Create an instance of VideoDataset
video_dataset = VideoDataset(sample_vids, labels_path)

# Access a sample data point
sample_index = 0
sample_data = video_dataset[sample_index]
print(sample_data)
