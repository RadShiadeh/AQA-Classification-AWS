from moviepy.editor import VideoFileClip
import os
import json

def get_vid_length(path):
    clip = VideoFileClip(path)
    return clip.duration

def create_data(data_dir, out):
    durations = {}

    for file in os.listdir(data_dir):
        if file.endswith(".mp4"):
            vid_path = os.path.join(data_dir, file)
            duration = get_vid_length(vid_path)
            durations[file] = duration

    with open(out, 'w') as f:
        json.dump(durations, f, indent=4)



dir = "../dissData/allVids"
out_path = "./labels/video_durations.json"

create_data(dir, out_path)