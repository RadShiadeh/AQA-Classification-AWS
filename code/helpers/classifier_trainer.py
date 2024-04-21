import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import ClassifierCNN3D
from dataloader_npy import VideoDataset

print("starting")

step = 0
log_frequency = 5
running_loss_print_freq = 50

def print_metrics(epoch, loss, accuracy_class, data_load_time, step_time, loss_type):
    epoch_step = step % len(video_dataset)
    print(
        f"epoch: [{epoch}], "
        f"step: [{epoch_step}/{len(video_dataset)}], "
        f"batch loss: {loss:.5f}, "
        f"accuracy_class: {accuracy_class:.5f}, "
        f"data load time: {data_load_time:.5f}, "
        f"step time: {step_time:.5f}",
        f"for {loss_type}"
    )

def log_metrics(epoch, loss, accuracy_class, data_load_time, step_time):
    summary_writer.add_scalar("epoch", epoch, step)
    summary_writer.add_scalar(
        "loss",
        {"train": float(loss.item())},
        step
    )
    summary_writer.add_scalar(
        "accuracy_class",
        accuracy_class,
        step
    )
    summary_writer.add_scalar(
        "time/data", data_load_time, step
    )
    summary_writer.add_scalar(
        "time/data", step_time, step
    )

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print_frequency = 20

classifier = ClassifierCNN3D()

labels_path = "../labels/train_labels/train.pkl"
sample_vids = "../../dissData/video_npy/train"
video_dataset = VideoDataset(sample_vids, labels_path, transform=None, num_frames=64)

batch_size = 16

data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

classifier = classifier.to(device)

criterion_classification = nn.BCELoss()

optimizer = optim.AdamW(classifier.parameters(), lr=0.0001)

summary_writer = SummaryWriter()

num_epochs = 50
print("loaded all models, going into training loop")
for epoch in range(num_epochs):
    print('-------------------------------------------------------------------------------------------------------')
    data_load_start_time = time.time()
    classification_running_loss = 0.0
    total_samples = 0
    correct_predictions_class = 0
    threshold = 0.5
    for _, batch_data in enumerate(data_loader):
        if epoch == 1:
            break
        frames = batch_data[0].type(torch.FloatTensor).to(device)
        frames = frames.permute(0, 4, 1, 2, 3)
        classification_labels = batch_data[1].type(torch.FloatTensor).to(device)

        data_load_end_time = time.time()

        classifier.train()
        
        optimizer.zero_grad()
            
        outputs = classifier(frames)
        classification_output = outputs.squeeze()

        classification_loss = criterion_classification(classification_output, classification_labels)
        
        classification_loss.backward()

        optimizer.step()

        classification_running_loss += classification_loss.item()

        # Compute accuracy
        predicted = torch.round(classification_output)
        correct_predictions_class += (predicted == classification_labels).sum().item()
        total_samples += classification_labels.size(0)

        if ((step+1) % running_loss_print_freq) == 0:
            print(f"average running loss per mini batch of classification loss: {classification_running_loss / batch_size:.3f} at [epoch, step]: {[epoch+1, step+1]}")

        data_load_time = data_load_end_time - data_load_start_time
        step_time = time.time() - data_load_end_time
        
        # Compute accuracy
        accuracy_class = correct_predictions_class / total_samples
        
        if ((step + 1) % log_frequency) == 0:
            log_metrics(epoch, classification_loss, accuracy_class, data_load_time, step_time)
            
        if ((step + 1) % print_frequency) == 0:
            print_metrics(epoch+1, classification_loss, accuracy_class, data_load_time, step_time, "classification")

        step += 1
    
    #import sys; sys.exit()
    

    if (epoch + 1) % 10 == 0:
        torch.save(classifier.state_dict(), 'ClassifierCNN3D_model.pth')
    


summary_writer.close()
