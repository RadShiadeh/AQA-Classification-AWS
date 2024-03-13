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
video_dataset = VideoDataset(sample_vids, labels_path, transform=None)

sample_index = 0
sample_data = video_dataset[sample_index]
sample_frames = sample_data['video']
sample_label = {
    'classification': sample_data['classification'],
    'score': sample_data['score']
}

fc = FullyConnected()
score_reg = ScoreRegressor()

data_loader = DataLoader(video_dataset, batch_size=5, shuffle=True)

eteModel = EndToEndModel(classifier, fc, final_score_regressor=score_reg)

criterion = nn.BCELoss()
optimizer = optim.AdamW(eteModel.parameters(), lr=0.00001)

# SummaryWriter setup
summary_writer = SummaryWriter()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, batch_data in enumerate(data_loader):
        frames = batch_data['video'].to(device)
        frames = frames.permute(0, 2, 1, 3, 4)
        
        classification_labels = batch_data['classification'].to(device)
        score_labels = batch_data['score'].to(device)
        
        optimizer.zero_grad()

        eteModel = eteModel.to(device)
    
        outputs = eteModel(frames)
        classification_output = outputs['classification']
        final_score_output = outputs['final_score']

        classification_loss = criterion(classification_output, classification_labels.float().view(-1, 1))
        final_score_loss = criterion(final_score_output, score_labels.float().view(-1, 1))

        loss = classification_loss + final_score_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        global_step = epoch * len(data_loader) + batch_idx
        summary_writer.add_scalar('Loss', loss.item(), global_step)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Close the SummaryWriter when done
summary_writer.close()
