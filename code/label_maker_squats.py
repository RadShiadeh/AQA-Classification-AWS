import json

labels_path_knee_in = "Squat/Labeled_Dataset/Labels/error_knees_inward.json"
labels_path_knee_out = "Squat/Labeled_Dataset/Labels/error_knees_forward.json"
video_durations = "labels/video_durations.json"

with open(labels_path_knee_in, 'r') as f:
    data_knee_in = json.load(f)

with open(labels_path_knee_out, 'r') as f1:
    data_knee_out = json.load(f1)

with open(video_durations, 'r') as d:
    durations = json.load(d)

scores = {}

for id, errors in data_knee_in.items():
    if errors == []:
        scores[id] = 5
    else:
        for e in errors:
            err_dur = e[1] - e[0]
            dur = durations[id]
            score = ((dur - err_dur) * 5) / dur
            scores[id] = round(score, 2)


for id, errors in data_knee_out.items():
    if errors == []:
        scores[id] += 5
    else:
        for e in errors:
            err_dur = e[1] - e[0]
            dur = durations[id]
            score = ((dur - err_dur) * 5) / dur
            scores[id] += round(score, 2)

out_path = "labels/squats_aqa.json"
with open(out_path, 'w') as d:
    json.dump(scores, d, indent=4)