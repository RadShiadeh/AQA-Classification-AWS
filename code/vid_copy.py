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


# Function Definition (copy_videos)

# Parameters:
# source_folder: The folder containing the original video files.
# destination_folder: The folder where the videos will be copied.
# data: A dictionary mapping video IDs to their associated information.
# Functionality:
# Checks if the destination folder exists; if not, it creates the folder.
# Iterates through each video ID in the provided dictionary.
# Constructs the source and destination paths for each video.
# Uses shutil.copyfile to copy the video file from the source to the destination.
# Loading Train Data from JSON

# The script loads the training data from a JSON file (../../dissData/valid_labels/valid_data.json). This file likely contains information about the validation set, including video IDs.
# Setting Source and Destination Folders

# Defines the source folder (../../dissData/allVids) containing the original video files.
# Defines the destination folder (../../dissData/valid_vids) where the videos will be copied.
# Calling copy_videos Function

# Invokes the copy_videos function with the source folder, destination folder, and the loaded validation data.