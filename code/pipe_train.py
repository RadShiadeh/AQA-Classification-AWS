import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pipe_models import BinaryClassifier, OverHeadPressAQA, BarbellSquatsAQA
from dataloader_pipe import VideoDataset
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def evaluate_scorer(scorer, data_loader, device):
    scorer.eval()
    predicted_scores = []
    true_scores = []
    with torch.no_grad():
        for batch_data in data_loader:
            frames = batch_data[0].type(torch.FloatTensor).to(device)
            frames = frames.permute(0, 4, 1, 2, 3)
            score_labels = batch_data[1].type(torch.FloatTensor).to(device)


            output = scorer(frames)

            predicted_scores.extend(output.to(device).numpy())
            true_scores.extend(score_labels.to(device).numpy())

    predicted_scores = np.array(predicted_scores)
    true_scores = np.array(true_scores)

    scorer.train()

    return predicted_scores, true_scores

def auc_classifier(classifier, test_data, device):
    classifier.eval()

    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for data in test_data:
            frames = data[0].type(torch.FloatTensor).to(device)
            frames = frames.permute(0, 4, 1, 2, 3)
            classification_labels = data[1].type(torch.FloatTensor).to(device)

            output = classifier(frames)

            true_labels.extend(classification_labels.to(device).numpy())
            predicted_probs.extend(output.to(device).numpy())
        
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    auc_score = roc_auc_score(true_labels, predicted_probs)

    classifier.train()
    
    return auc_score




def print_metrics(epoch, loss, type, epoch_end, acc=0.0):
    print(
        f"epoch: [{epoch}], "
        f"batch loss: {loss:.3f}, "
        f"model type: {type}"
        f"epoch end time: {epoch_end:.3f}, "
        f"acc: {acc:.3f}"
    )


def log_metrics(epoch, loss, data_load_time, step_time, summary_writer, step):
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

def train_classifier(num_epochs, classifier, class_train_data_loader, optimizer, eval_freq, class_test_data_loader, print_frequency, device,
                      log_frequency, criterion_classification, summary_writer, step):
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('-------------------------------------------------------------------------------------------------------')
        data_load_start_time = time.time()
        classification_running_loss = 0.0
        classifier.train()

        for _, batch_data in enumerate(class_train_data_loader):
            frames = batch_data[0].type(torch.FloatTensor).to(device)
            frames = frames.permute(0, 4, 1, 2, 3)
            
            classification_labels = batch_data[1].type(torch.FloatTensor).to(device)

            data_load_end_time = time.time()

            optimizer.zero_grad()

            output_class = classifier(frames)

            classification_loss = criterion_classification(output_class, classification_labels.float().view(-1, 1))

            loss = 0
            loss += classification_loss

            loss.backward()
            optimizer.step()

            classification_running_loss += classification_loss.item()

            data_load_time = data_load_end_time - data_load_start_time
            step_time = time.time() - data_load_end_time

            if ((step + 1) % log_frequency) == 0:
                log_metrics(epoch, classification_loss, data_load_time, step_time, summary_writer, step)

            step += 1
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if ((epoch + 1) % eval_freq) == 0:
            auc_class = auc_classifier(classifier, class_test_data_loader, device)
        
        avg_classification_loss = classification_running_loss / len(class_train_data_loader)

        if ((epoch + 1) % print_frequency) == 0:
            print_metrics(epoch=epoch+1, loss=avg_classification_loss, type="classification ", epoch_end=epoch_time, acc=auc_class)
            print(f"running loss classifier: {classification_running_loss:.3f}")

        if (epoch + 1) % 5 == 0:
            torch.save(classifier.state_dict(), 'classifier_model_r3d18.pth')


def train_aqa(num_epochs, scorer, train_data_loader, optimizer, eval_freq, test_data_loader, print_frequency, device,
                   log_frequency, criterion_scorer, criterion_scorer_penalty, summary_writer, step, model_type):
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('-------------------------------------------------------------------------------------------------------')
        data_load_start_time = time.time()
        scorer_running_loss = 0.0
        scorer.train()

        for _, batch_data in enumerate(train_data_loader):
            frames = batch_data[0].type(torch.FloatTensor).to(device)
            frames = frames.permute(0, 4, 1, 2, 3)
            
            score_labels = batch_data[1].type(torch.FloatTensor).to(device)
            score_labels = score_labels.float().view(-1, 1)

            data_load_end_time = time.time()

            optimizer.zero_grad()

            output_score = scorer(frames)

            final_score_loss = criterion_scorer(output_score, score_labels.float()) + criterion_scorer_penalty(output_score, score_labels.float())

            loss = 0
            loss += final_score_loss

            loss.backward()
            optimizer.step()

            scorer_running_loss += final_score_loss.item()

            data_load_time = data_load_end_time - data_load_start_time
            step_time = time.time() - data_load_end_time

            if ((step + 1) % log_frequency) == 0:
                log_metrics(epoch, final_score_loss, data_load_time, step_time, summary_writer, step)

            step += 1
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if ((epoch + 1) % eval_freq) == 0:
            pred_score, true_score = evaluate_scorer(scorer, test_data_loader, device)
            correlation_coeff, _ = spearmanr(pred_score, true_score)
        
        avg_scorer_loss = scorer_running_loss / len(train_data_loader)

        if ((epoch + 1) % print_frequency) == 0:
            scorer_type = "spearman score for " + model_type + " "
            print_metrics(epoch=epoch+1, loss=avg_scorer_loss, type=scorer_type, epoch_end=epoch_time, acc=correlation_coeff)
            print(f"running loss avg scorer ohp: {avg_scorer_loss:.3f}")

        if (epoch + 1) % 5 == 0:
            path_name = "scorer_" + model_type + "_model_r3d18.path"
            torch.save(scorer.state_dict(), path_name)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    step = 0
    log_frequency = 5
    running_loss_print_freq = 50
    print_frequency = 1
    batch_size = 16
    eval_freq = 1

    all_vids_path = "../../dissData/video_npy_reduced/allVids"

    #ohp aqa labels
    ohp_aqa_train_labels = "../labels/ohp_aqa_labels/ohp_aqa_train.pkl"
    video_dataset_ohp_train = VideoDataset(all_vids_path, ohp_aqa_train_labels, transform=None, num_frames=32)

    ohp_aqa_labels_valid = "../labels/ohp_aqa_labels/ohp_aqa_valid.pkl"
    video_dataset_valid_ohp = VideoDataset(all_vids_path, ohp_aqa_labels_valid, transform=None, num_frames=32)

    labels_test_ohp = "../labels/ohp_aqa_labels/ohp_aqa_test.pkl"
    video_dataset_test_ohp = VideoDataset(all_vids_path, labels_test_ohp, transform=None, num_frames=32)

    ohp_train_data_loader = DataLoader(video_dataset_ohp_train, batch_size=batch_size, shuffle=True)
    ohp_validation_data = DataLoader(video_dataset_valid_ohp, batch_size)
    ohp_test_data_loader = DataLoader(video_dataset_test_ohp, batch_size)


    #squats aqa labels
    squats_aqa_train_labels = "../labels/squat_aqa_labels/squats_aqa_train.pkl"
    video_dataset_squats_train = VideoDataset(all_vids_path, squats_aqa_train_labels, transform=None, num_frames=32)

    squats_aqa_labels_valid = "../labels/squat_aqa_labels/squats_aqa_valid.pkl"
    video_dataset_valid_squats = VideoDataset(all_vids_path, squats_aqa_labels_valid, transform=None, num_frames=32)

    labels_test_squats = "../labels/squat_aqa_labels/squats_aqa_test.pkl"
    video_dataset_test_squats = VideoDataset(all_vids_path, labels_test_squats, transform=None, num_frames=32)

    squat_train_data_loader = DataLoader(video_dataset_squats_train, batch_size=batch_size, shuffle=True)
    squat_validation_data = DataLoader(video_dataset_valid_squats, batch_size)
    squat_test_data_loader = DataLoader(video_dataset_test_squats, batch_size)


    #classification labels
    class_train_labels = "../labels/classification_labels/class_labels_train.pkl"
    video_dataset_class_train = VideoDataset(all_vids_path, class_train_labels, transform=None, num_frames=32)

    class_labels_valid = "../labels/classification_labels/class_labels_valid.pkl"
    video_dataset_valid_class = VideoDataset(all_vids_path, class_labels_valid, transform=None, num_frames=32)

    labels_test_class = "../labels/classification_labels/class_labels_test.pkl"
    video_dataset_test_class = VideoDataset(all_vids_path, labels_test_class, transform=None, num_frames=32)

    class_train_data_loader = DataLoader(video_dataset_class_train, batch_size=batch_size, shuffle=True)
    class_validation_data = DataLoader(video_dataset_valid_class, batch_size)
    class_test_data_loader = DataLoader(video_dataset_test_class, batch_size)


    classifier = BinaryClassifier()

    over_head_press_AQA = OverHeadPressAQA()
    barbell_squat_AQA = BarbellSquatsAQA()

    classifier.to(device)
    over_head_press_AQA.to(device)
    barbell_squat_AQA.to(device)


    criterion_classification = nn.BCELoss()
    criterion_scorer = nn.MSELoss()
    criterion_scorer_penalty = nn.L1Loss()

    optimizer_class = optim.AdamW(classifier.parameters(), lr=1e-4)
    optimizer_ohp = optim.AdamW(over_head_press_AQA.parameters(), lr=1e-4)
    optimizer_squats = optim.AdamW(barbell_squat_AQA.parameters(), lr=1e-4)


    summary_writer = SummaryWriter()
    num_epochs = 50

    #classifier trainer
    train_classifier(num_epochs, classifier, class_train_data_loader, optimizer_class, eval_freq, 
                     class_test_data_loader, print_frequency, device, log_frequency,
                     criterion_classification, summary_writer, step)

    #ohp aqa trainer
    train_aqa(num_epochs, over_head_press_AQA, ohp_train_data_loader, optimizer_ohp, eval_freq, ohp_test_data_loader,
                  print_frequency, device, log_frequency, criterion_scorer, criterion_scorer_penalty, summary_writer, step, "ohp")
    
    #squat aqa trainer
    train_aqa(num_epochs, barbell_squat_AQA, squat_train_data_loader, optimizer_squats, eval_freq, squat_test_data_loader,
                  print_frequency, device, log_frequency, criterion_scorer, criterion_scorer_penalty, summary_writer, step, "squat")

    summary_writer.close()

if __name__ == "__main__":
    main()