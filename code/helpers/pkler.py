import pickle
import json

def convert_json_to_pkl(json_file, pkl_file):
    count_squat = 0
    count_ohp = 0

    with open(json_file, 'r') as f:
        data = json.load(f)

    reduced = {}
    for id, v in data.items():
        if v[0] == 1 and count_squat <= 50:
            count_squat += 1
            reduced[id] = v
        elif v[0] == 0 and count_ohp <= 50:
            count_ohp += 1
            reduced[id] = v

        if len(reduced) >= 100:
            break
    
    print(len(reduced))
    
    with open(pkl_file, 'wb') as f:

        pickle.dump(reduced, f)

json_file = 'valid_labels.json'
pkl_file = 'labels_reduced_valid.pkl'
convert_json_to_pkl(json_file, pkl_file)