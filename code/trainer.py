import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import C3DC, FullyConnected, ScoreRegressor, EndToEndModel, ClassifierCNN3D
from dataLoader import VideoDataset


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

classifier = ClassifierCNN3D()

labels_path = "../labels/complete_labels.json"
sample_vids = "../../dissData/train_vids"
video_dataset = VideoDataset(sample_vids, labels_path, transform=None, resize_shape=(256, 256), num_frames=16)

cnnLayer = C3DC()
fc = FullyConnected()
score_reg = ScoreRegressor()
fc = fc.to(device)
score_reg = score_reg.to(device)
batch_size = 10

data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: [data for data in batch if data is not None])

eteModel = EndToEndModel(classifier, cnnLayer, fc, final_score_regressor=score_reg)
eteModel = eteModel.to(device)

criterion_classification = nn.BCELoss()
criterion_scorer = nn.CrossEntropyLoss()
optimizer = optim.AdamW(eteModel.parameters(), lr=0.00001)

# SummaryWriter setup
summary_writer = SummaryWriter()

# Training loop
num_epochs = batch_size
for epoch in range(num_epochs):
    print("in epoch")
    for iteration, batch_data in enumerate(data_loader):
        print("in dataloader")
        for data in batch_data:
            print("made it here")
            frames = data['video'].to(device)
            frames = frames.permute(0, 2, 1, 3, 4)

            classification_labels = data['classification'].to(device)
            score_labels = data['score'].to(device)
            
            optimizer.zero_grad()

            print("loaded everything, training now")
            outputs = eteModel(frames)
            print("got outs")
            classification_output = outputs['classification']
            final_score_output = outputs['final_score']

            final_score_output = torch.sigmoid(final_score_output)
            classification_output = torch.sigmoid(classification_output)

            classification_loss = criterion_classification(classification_output, classification_labels.float().view(-1, 1))
            final_score_loss = criterion_scorer(final_score_output, score_labels.float().view(-1, 1))
            print("got final res")

            loss = classification_loss + final_score_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log loss to TensorBoard
            global_step = epoch * 380 + iteration
            summary_writer.add_scalar('Loss', loss.item(), global_step)
            
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Iteration {iteration + 1}, Loss: {loss.item()}')

# Close the SummaryWriter when done
summary_writer.close()