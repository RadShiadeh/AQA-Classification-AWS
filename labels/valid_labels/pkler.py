import pickle
import json

def convert_json_to_pkl(json_file, pkl_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

def dump_complete_labels(complete_path_json, partition_path_json, dump_path):
    with open(partition_path_json, 'r') as d:
        data_partition = json.load(d)
    
    partition_keys = list(data_partition.keys())

    with open(complete_path_json, 'r') as f:
        data_complete = json.load(f)
    

    res = {}
    for k, v in data_complete.items():
        if k in partition_keys:
            res[k] = v
    
    with open(dump_path, 'w') as w:
        json.dump(res, w, indent=4)

def test(path, path_org):
    with open(path, "r") as file:
        data = json.load(file)

    keys = list(data.keys())
    
    with open(path_org, 'r') as f:
        data_o = json.load(f)

    keys_o = list(data_o.keys())

    if len(keys) != len(keys_o):
        print("wrong")
        import sys; sys.exit(0)
    else:
        return True



def main():
    valid_labels_path = "./valid_data.json"
    test_labels_path  = "../test_labels/test_data.json"
    train_labels_path = "../train_labels/train_labels.json"

    valid_dump = "./valid_labels.json"
    test_dump   = "../test_labels/test_labels.json"
    train_dump = "../train_labels/train_data.json"
    
    valid_pkl = "./valid.pkl"
    test_pkl = "../test_labels/test.pkl"
    train_pkl = "../train_labels/train.pkl"

    complete_path = "../complete_labels.json"

    dump_complete_labels(complete_path, valid_labels_path, valid_dump)
    test(valid_dump, valid_labels_path)

    dump_complete_labels(complete_path, train_labels_path, train_dump)
    test(train_dump, train_labels_path)

    dump_complete_labels(complete_path, test_labels_path, test_dump)
    test(test_dump, test_labels_path)

    convert_json_to_pkl(valid_dump, valid_pkl)
    convert_json_to_pkl(test_dump, test_pkl)
    convert_json_to_pkl(train_dump, train_pkl)

if __name__=="__main__":
    main()