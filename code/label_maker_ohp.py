import json

data_path = "OHP/Labeled_Dataset/Labels/error_knees.json"
vid_duration_path = "labels/video_durations.json"

with open(data_path, 'r') as d:
    file = json.load(d)

with open(vid_duration_path, 'r') as dur:
    durations = json.load(dur)


scores = {}
for ids, labels in file.items():
    if labels == []:
        scores[ids] = 10
    else:
        for l in labels:
            tmp_error_dur = l[1] - l[0]
            total_durr = durations[ids]
            score = ((total_durr - tmp_error_dur) * 10) / total_durr
            scores[ids] = round(score, 1)



out = "./labels/OHP_Aqa.json"
with open(out, 'w') as f:
    json.dump(scores, f, indent=4)

# Read the error data from the file error_knees.json.
# Read the video durations data from the file video_durations.json.
# Iterate through each video ID in the error data.
# If there are no errors (empty list), assign a perfect score of 10.
# If there are errors, calculate the error duration, total video duration, and score based on the formula provided. Round the score to one decimal place.
# Store the calculated score for each video ID in the perfect dictionary.
# Write the perfect dictionary to a new JSON file named OHP_Aqa.json with proper indentation