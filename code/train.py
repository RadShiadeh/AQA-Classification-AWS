import dataloader
from torch.utils.data import DataLoader



video_path = "../../dissData/allVids"
labels_path = "../../dissData/labels/all_labels.json"
loader = dataloader.VideoDataset(video_path, labels_path, transform=None)

data = DataLoader(loader, batch_size=4, shuffle=True)