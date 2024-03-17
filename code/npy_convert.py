import cv2
import numpy as np
import os
import json
from tester import *

def convert_video_to_npy(video_file, output_folder, num_frames=128, resize_shape=(256, 256)):
    cap = cv2.VideoCapture(video_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(video_file)
    filename_without_extension = os.path.splitext(filename)[0]

    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_shape)
        frames.append(frame)
        frame_count += 1

    # Pad frames if less than num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros_like(frames[0]))

    frames_array = np.array(frames)
    np.save(os.path.join(output_folder, f"{filename_without_extension}.npy"), frames_array)

    cap.release()

def make_npy(vids_path, out_folder_path, labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    for video_id, _ in labels.items():
        video_path = os.path.join(vids_path, f"{video_id}.mp4")
        convert_video_to_npy(video_path, out_folder_path)

def main():
    train_vids = "../../dissData/train_vids"
    test_vids = "../../dissData/test_vids"
    valid_vids = "../../dissData/valid_vids"

    test_out_folder = "./test"
    train_out_folder = "./train"
    valid_out_folder = "./valid"

    train_labels_path = "./labels/train_labels/train_labels.json"
    test_labels_path = "./labels/test_labels/test_data.json"
    valid_labels_path = "./labels/valid_labels/valid_data.json"

    make_npy(train_vids, train_out_folder, train_labels_path)
    make_npy(test_vids, test_out_folder, test_labels_path)
    make_npy(valid_vids, valid_out_folder, valid_labels_path)

    fuck(valid_out_folder, test_out_folder, train_out_folder)

if __name__ == "__main__":
    main()