import os
import shutil
import json

def copy_videos(source_folder, destination_folder, data):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for video_id in data.keys():
        source_path = os.path.join(source_folder, f"{video_id}.mp4")
        destination_path = os.path.join(destination_folder, f"{video_id}.mp4")

        shutil.copyfile(source_path, destination_path)

train_json_path = '../../dissData/valid_labels/valid_data.json'
with open(train_json_path, 'r') as f:
    train_data = json.load(f)

source_videos_folder = '../../dissData/allVids'
destination_folder = '../../dissData/valid_vids'

copy_videos(source_videos_folder, destination_folder, train_data)