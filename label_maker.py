import json
import os

data_path = "OHP/Labeled_Dataset/Labels/error_knees.json"

with open(data_path, 'r') as d:
    file = json.load(d)

ids = file.keys()
labels = file.values()


def findmin(labels):
    res = 99999
    for label in labels:
        for l in label:
            tmp = l[1] - l[0]
            res = min(res, tmp)

    
    return res

min_second_val = findmin(labels)

def findmax(labels):
    res = -1
    for label in labels:
        for l in label:
            tmp = l[1] - l[0]
            res = max(res, tmp)
    
    return res

max_seconds_val = findmax(labels)

print(max_seconds_val)
print(min_second_val)