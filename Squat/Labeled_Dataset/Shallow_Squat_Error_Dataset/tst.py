import json

path = "labels_shallow_depth.json"

with open(path, 'r') as f:
    data = json.load(f)

print(len(data))