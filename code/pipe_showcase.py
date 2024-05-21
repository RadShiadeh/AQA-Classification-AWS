import torch
from pipe_models import BinaryClassifier, OverHeadPressAQA, BarbellSquatsAQA
import cv2
import numpy as np

class Showcase:
    def __init__(self, video_path, classifier_path, ohp_aqa_model_path, squats_aqa_model_path, device):
        self.video_path = video_path
        self.classifier_path = classifier_path
        self.ohp_aqa_model_path = ohp_aqa_model_path
        self.squats_aqa_model_path = squats_aqa_model_path
        self.device = device

    def video_to_frames(self, video_path, num_frames=32):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))  # Resize to match the model's expected input size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) < num_frames:
            frames.extend([np.zeros_like(frames[0]) for _ in range(num_frames - len(frames))])
        elif len(frames) > num_frames:
            frames = frames[:num_frames]
        
        frames = np.array(frames)
        frames = frames.astype(np.float32) / 255.0  # Normalize frames
        frames = frames.transpose(3, 0, 1, 2)  # Change to (C, D, H, W)
        return torch.tensor(frames).unsqueeze(0)  # Add batch dimension

    def load_model(self, model_path, model_class):
        model = model_class().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def res(self):
        classifier = self.load_model(self.classifier_path, BinaryClassifier)
        ohp_aqa = self.load_model(self.ohp_aqa_model_path, OverHeadPressAQA)
        squats_aqa = self.load_model(self.squats_aqa_model_path, BarbellSquatsAQA)

        frames = self.video_to_frames(self.video_path).to(self.device)

        with torch.no_grad():
            classification_output = classifier(frames)
            classification = int(classification_output >= 0.5)
        
        if classification == 1:
            with torch.no_grad():
                score = ohp_aqa(frames).item()
        else:
            with torch.no_grad():
                score = squats_aqa(frames).item()
        
        return {"classification": classification, "score": score}

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_vid = "path/to/your/video.mp4"
    classifier_weights = "path/to/classifier_model.pth"
    ohp_weights = "path/to/ohp_aqa_model.pth"
    squats_weights = "path/to/squats_aqa_model.pth"

    showcase = Showcase(input_vid, classifier_weights, ohp_weights, squats_weights, device)

    res = showcase.res()
    
    print(res)

if __name__ == "__main__":
    main()
