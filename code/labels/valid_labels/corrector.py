import json
import pickle

def correct(path):
    with open(path, 'r') as d:
        data = json.load(d)

    
    for k, v in data.items():
        if v[1] > 10:
            data[k] = [v[0], 10]
        elif v[1] < 0:
            data[k] = [v[0], 0]
    
    with open(path, 'w') as w:
        json.dump(data, w, indent=4)

def convert_json_to_pkl(json_file, pkl_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

valid_dump = "./valid_labels.json"
test_dump   = "../test_labels/test_labels.json"
train_dump = "../train_labels/train_data.json"

valid_pkl = "./valid.pkl"
test_pkl = "../test_labels/test.pkl"
train_pkl = "../train_labels/train.pkl"

correct(valid_dump)
correct(test_dump)
correct(train_dump)

convert_json_to_pkl(valid_dump, valid_pkl)
convert_json_to_pkl(test_dump, test_pkl)
convert_json_to_pkl(train_dump, train_pkl)