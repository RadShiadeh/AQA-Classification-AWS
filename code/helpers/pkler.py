import pickle
import json

def convert_json_to_pkl(json_file, pkl_file):
    count_squat = 0
    count_ohp = 0

    with open(json_file, 'r') as f:
        data = json.load(f)

    reduced = {}
    for id, v in enumerate(data):
        if v == 1:
            count_squat += 1
        else:
            count_ohp += 1

        reduced[id] = v
        if count_ohp >= 500 and count_squat >= 500:
            break

    with open(pkl_file, 'wb') as f:

        pickle.dump(reduced, f)

json_file = 'train_labels.json'
pkl_file = 'labels_reduced.pkl'
convert_json_to_pkl(json_file, pkl_file)