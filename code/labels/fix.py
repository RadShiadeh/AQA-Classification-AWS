import pickle
import json
import random

def split_data(data_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    #assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of the ratios must be 1."
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    data_items = list(data.items())
    random.shuffle(data_items)  

    total_size = len(data_items)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = dict(data_items[:train_size])
    val_data = dict(data_items[train_size:train_size + val_size])
    test_data = dict(data_items[train_size + val_size:])

    return train_data, val_data, test_data


def convert_json_to_pkl(data, pkl_file_train, pkl_file_test, pkl_file_valid):
    train, test, valid = split_data(data)
        
    with open(pkl_file_train, 'wb') as f:
        pickle.dump(train, f)
    
    with open(pkl_file_test, 'wb') as f:
        pickle.dump(test, f)
    
    with open(pkl_file_valid, 'wb') as f:
        pickle.dump(valid, f)

json_file = 'squats_aqa.json'
pkl_file_train = 'squats_aqa_train.pkl'
pkl_file_test = 'squats_aqa_test.pkl'
pkl_file_valid = 'squats_aqa_valid.pkl'
convert_json_to_pkl(json_file, pkl_file_train, pkl_file_test, pkl_file_valid)