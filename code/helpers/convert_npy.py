import cv2
import numpy as np
import os
import json

def convert_video_to_npy(video_file, output_folder, num_frames=16, resize_shape=(256, 256)):
    cap = cv2.VideoCapture(video_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = os.path.basename(video_file)
    filename_without_extension = os.path.splitext(filename)[0]

    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_shape)

        frames.append(frame)

        if len(frames) == num_frames or not ret:
            while len(frames) < num_frames:
                frames.append(np.zeros_like(frames[0]))

            frames_array = np.array(frames)

            np.save(os.path.join(output_folder, f"{filename_without_extension}_{frame_count}.npy"), frames_array)

            frames = []
            frame_count += 1

    cap.release()

def make_npy(vids_path, out_folder_path, lables_path):
    with open(lables_path, 'r') as f:
        labels_ = json.load(f)
    
    video_ids_dict = labels_.keys()
    video_ids = []
    for key in video_ids_dict:
        video_ids.append(key)
    
    for id in video_ids:
        single_video_path = os.path.join(vids_path, f"{id}.mp4")
        convert_video_to_npy(single_video_path, out_folder_path)

def main():
    train_vids = "../train_vids"
    test_vids = "../test_vids"
    valid_vids = "../valid_vids"

    test_out_folder = "./test"
    train_out_folder = "./train"
    valid_out_folder = "./valid"

    train_labels_path = "../labels/train_labels.json"
    test_labels_path = "../labels/test_data.json"
    valid_labels_path = "../labels/valid_data.json"

    make_npy(train_vids, train_out_folder, train_labels_path)
    make_npy(test_vids, test_out_folder, test_labels_path)
    make_npy(valid_vids, valid_out_folder, valid_labels_path)

if __name__ == "__main__":
    main()