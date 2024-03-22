import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import C3DC, FullyConnected, ScoreRegressor, EndToEndModel, ClassifierCNN3D
from dataloader_npy import VideoDataset
import numpy as np
from scipy.stats import spearmanr


def evaluate_model(model, test_data_loader):
    model.eval()
    predicted_scores = []
    true_scores = []

    with torch.no_grad():
        for batch_data in test_data_loader:
            frames = batch_data[0].type(torch.FloatTensor).to(device)
            frames = frames.permute(0, 4, 1, 2, 3)
            score_labels = batch_data[2].type(torch.FloatTensor).to(device)

            outputs = model(frames)
            final_score_output = outputs['final_score']

            predicted_scores.extend(final_score_output.cpu().numpy())
            true_scores.extend(score_labels.cpu().numpy())

    predicted_scores = np.array(predicted_scores)
    true_scores = np.array(true_scores)

    return predicted_scores, true_scores

def print_metrics(epoch, loss, accuracy, type):
        epoch_step = step % len(video_dataset)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(video_dataset)}], "
                f"batch loss: {loss:.5f}, "
                f"accuracy: {accuracy}, "
                f"data load time: "
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
eval_freq = 1

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
cnnLayer = C3DC()
fc = FullyConnected()
score_reg = ScoreRegressor()
eteModel = EndToEndModel(classifier, cnnLayer, fc, final_score_regressor=score_reg)

fc = fc.to(device)
score_reg = score_reg.to(device)
cnnLayer = cnnLayer.to(device)
classifier = classifier.to(device)
eteModel = eteModel.to(device)


criterion_classification = nn.BCELoss()
criterion_scorer = nn.CrossEntropyLoss()


optimizer_classifier = optim.AdamW(classifier.parameters(), lr=0.0005)
optim_cnn = optim.AdamW(cnnLayer.parameters(), lr=0.0005)
optim_scor_reg = optim.AdamW(score_reg.parameters(), lr=0.0005)
optim_fc = optim.AdamW(fc.parameters(), lr=0.0005)


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
    eteModel.train()
    for _, batch_data in enumerate(validation_data):
        frames = batch_data[0].type(torch.FloatTensor).to(device)
        frames = frames.permute(0, 4, 1, 2, 3)
        classification_labels = batch_data[1].type(torch.FloatTensor).to(device)
        score_labels = batch_data[2].type(torch.FloatTensor).to(device)

        data_load_end_time = time.time()
            
        outputs = eteModel(frames)
        classification_output = outputs['classification']
        final_score_output = outputs['final_score']

        classification_loss = criterion_classification(classification_output, classification_labels.float().view(-1, 1))
        final_score_loss = criterion_scorer(final_score_output, score_labels.float().view(-1, 1))

        classification_loss.backward()
        final_score_loss.backward()

        optimizer_classifier.step()
        optim_cnn.step()
        optim_fc.step()
        optim_scor_reg.step()
        
        optimizer_classifier.zero_grad()
        optim_cnn.zero_grad()
        optim_scor_reg.zero_grad()
        optim_fc.zero_grad()

        classification_running_loss += classification_loss.item()
        scorer_running_loss += final_score_loss.item()

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

        if ((step + 1) % log_frequency) == 0:
            log_metrics(epoch, classification_loss, data_load_time, step_time)
            log_metrics(epoch, final_score_loss, data_load_time, step_time)

        step += 1

    if ((epoch + 1) % eval_freq) == 0:
        pred_score, true_score = evaluate_model(eteModel, test_data_loader)
        correlation_coeff, _ = spearmanr(pred_score, true_score)

    if ((epoch + 1) % print_frequency) == 0:
        print_metrics(epoch=epoch+1, loss=classification_loss, accuracy=accuracy_class, type="classification")
        print_metrics(epoch=epoch+1, loss=final_score_loss, accuracy=correlation_coeff, type="scorer spearmanr correlation")

    if (epoch + 1) % 5 == 0:
        torch.save(eteModel.state_dict(), 'ETE_model.pth')
    


summary_writer.close()
