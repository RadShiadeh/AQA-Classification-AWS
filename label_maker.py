import json

data_path = "OHP/Labeled_Dataset/Labels/error_knees.json"
vid_duration_path = "labels/video_durations.json"

with open(data_path, 'r') as d:
    file = json.load(d)

with open(vid_duration_path, 'r') as dur:
    durations = json.load(dur)


perfect = {}
for ids, labels in file.items():
    if labels == []:
        perfect[ids] = 10
    else:
        for d in durations.values():
            tmp_dur = durations[ids]
            print(d)
            print(tmp_dur)



# out = "./labels/OHP_Aqa.json"
# with open(out, 'w') as f:
#     json.dump(perfect, f, indent=4)