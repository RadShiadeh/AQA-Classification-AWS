import pickle
import json

def convert_json_to_pkl(json_file, pkl_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

json_file = 'complete_labels.json'
pkl_file = 'labels.pkl'
convert_json_to_pkl(json_file, pkl_file)