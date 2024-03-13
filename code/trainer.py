import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import C3DC, FullyConnected, ScoreRegressor, EndToEndModel
from dataLoader import VideoDataset


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classifier = C3DC()

labels_path = "../labels/complete_labels.json"
sample_vids = "../../dissData/train_vids"
video_dataset = VideoDataset(sample_vids, labels_path, transform=None, resize_shape=(128, 128), num_frames=16)

fc = FullyConnected()
score_reg = ScoreRegressor()
fc = fc.to(device)
score_reg = score_reg.to(device)

data_loader = DataLoader(video_dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: [data for data in batch if data is not None])

eteModel = EndToEndModel(classifier, fc, final_score_regressor=score_reg)
eteModel = eteModel.to(device)

criterion = nn.BCELoss()
optimizer = optim.AdamW(eteModel.parameters(), lr=0.00001)

# SummaryWriter setup
summary_writer = SummaryWriter()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, batch_data in enumerate(data_loader):
        if batch_data is None:
            continue
        frames = batch_data[i]['video'].to(device)
        frames = frames.permute(0, 2, 1, 3, 4)
        print(frames.shape, f"at {i}")
        
        classification_labels = batch_data[i]['classification'].to(device)
        score_labels = batch_data[i]['score'].to(device)
        
        optimizer.zero_grad()

        
        outputs = eteModel(frames)
        classification_output = outputs['classification']
        final_score_output = outputs['final_score']

        final_score_output = torch.sigmoid(final_score_output)
        classification_output = torch.sigmoid(classification_output)

        classification_loss = criterion(classification_output, classification_labels.float().view(-1, 1))
        final_score_loss = criterion(final_score_output, score_labels.float().view(-1, 1))

        loss = classification_loss + final_score_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        global_step = epoch * len(data_loader) + i
        i += 1
        summary_writer.add_scalar('Loss', loss.item(), global_step)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Close the SummaryWriter when done
summary_writer.close()