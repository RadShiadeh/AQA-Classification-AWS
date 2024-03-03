import os
import shutil
import json

def copy_videos(source_folder, destination_folder, data):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for video_id in data.keys():
        source_path = os.path.join(source_folder, f"{video_id}.mp4")
        destination_path = os.path.join(destination_folder, f"{video_id}.mp4")

        # Copy the video file
        shutil.copyfile(source_path, destination_path)

# Load train_data.json
train_json_path = '../../dissData/valid_labels/valid_data.json'
with open(train_json_path, 'r') as f:
    train_data = json.load(f)

# Set the source and destination folders
source_videos_folder = '../../dissData/allVids'
destination_folder = '../../dissData/valid_vids'

# Copy videos to the destination folder
copy_videos(source_videos_folder, destination_folder, train_data)