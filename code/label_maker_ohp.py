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