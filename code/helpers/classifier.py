import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import C3DC, FullyConnected, ScoreRegressor, EndToEndModel, ClassifierCNN3D
from dataloader_npy import VideoDataset

def print_metrics(epoch, loss, data_load_time, step_time, accuracy, type):
        epoch_step = step % len(video_dataset)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(video_dataset)}], "
                f"batch loss: {loss:.5f}, "
                f"accuracy: {accuracy}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}",
                f"model type: {type}"
        )


def log_metrics(epoch, loss, data_load_time, step_time):
    summary_writer.add_scalar("epoch", epoch, step)
    summary_writer.add_scalars(
                "loss",
            {"train": float(loss.item())},
            step
    )
    summary_writer.add_scalar(
            "time/data", data_load_time, step
    )
    summary_writer.add_scalar(
            "time/data", step_time, step
    )

print("starting")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

step = 0
log_frequency = 5
running_loss_print_freq = 50
print_frequency = 1 #print after each epoch
batch_size = 16

train_labels_path = "../labels/train_labels/train.pkl"
train_vids = "../../dissData/video_npy/train"
video_dataset = VideoDataset(train_vids, train_labels_path, transform=None, num_frames=16)

labels_valid = "../labels/valid_labels/valid.pkl"
valid_vids = "../../dissData/video_npy/valid"
video_dataset_valid = VideoDataset(valid_vids, labels_valid, transform=None, num_frames=16)

labels_test = "../labels/valid_labels/valid.pkl"
test_vids = "../../dissData/video_npy/valid"
video_dataset_test = VideoDataset(test_vids, labels_test, transform=None, num_frames=16)

train_data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)
validation_data = DataLoader(video_dataset_valid, batch_size)
test_data_loader = DataLoader(video_dataset_test, batch_size)


classifier = ClassifierCNN3D()

classifier = classifier.to(device)


criterion_classification = nn.BCELoss()
criterion_scorer = nn.CrossEntropyLoss()

optimizer = optim.AdamW(classifier.parameters(), lr=0.0001)

summary_writer = SummaryWriter()

num_epochs = 50
print("loaded all models, going into training loop")
for epoch in range(num_epochs):
    print('-------------------------------------------------------------------------------------------------------')
    data_load_start_time = time.time()
    classification_running_loss = 0.0
    scorer_running_loss = 0.0
    total_samples = 0
    correct_predictions_class = 0
    total_samples_score = 0
    correct_score_predictions = 0
    threshold = 0.5
    classifier.train()
    for _, batch_data in enumerate(train_data_loader):
        frames = batch_data[0].type(torch.FloatTensor).to(device)
        frames = frames.permute(0, 4, 1, 2, 3)
        classification_labels = batch_data[1].type(torch.FloatTensor).to(device)

        data_load_end_time = time.time()
        
        optimizer.zero_grad()
            
        outputs = classifier(frames)
        classification_output = outputs['classification']

        classification_loss = criterion_classification(classification_output, classification_labels.float().view(-1, 1))
        classification_loss.backward()

        optimizer.step()

        classification_running_loss += classification_loss.item()

        # Compute accuracy
        _, predicted = torch.max(classification_output, 1)
        correct_predictions_class += (predicted == classification_labels).sum().item()
        total_samples += classification_labels.size(0)


        if ((step+1) % running_loss_print_freq) == 0:
            print(f"average running loss per mini batch of classification loss: {classification_running_loss / batch_size:.3f} at [epoch, step]: {[epoch+1, step+1]}")
            print(f"average running loss per mini batch of scorer loss: {scorer_running_loss / batch_size:.3f} at [epoch, step]: {[epoch+1, step+1]}")

        data_load_time = data_load_end_time - data_load_start_time
        step_time = time.time() - data_load_end_time
        
        accuracy_class = correct_predictions_class / total_samples
        accuracy_score = correct_score_predictions / total_samples_score

        if ((step + 1) % log_frequency) == 0:
            log_metrics(epoch, classification_loss, data_load_time, step_time)

        step += 1

    if ((epoch + 1) % print_frequency) == 0:
        print_metrics(epoch=epoch+1, loss=classification_loss, accuracy=accuracy_class, data_load_time=data_load_time, step_time=step_time, type="classification")

    if (epoch + 1) % 5 == 0:
        torch.save(classifier.state_dict(), 'ETE_model.pth')
    


summary_writer.close()
