import json

AQA_scores_path = "./labels/AQA_scores.json"
Class_labels_path = "./labels/all_labels.json"
out_path = "./labels/complete_labels.json"

with open(AQA_scores_path, 'r') as file:
    aqa_scores_data = json.load(file)

with open(Class_labels_path, 'r') as file:
    class_data = json.load(file)

# Create a new dictionary to hold the combined data
combined_scores_data = {}

# Iterate through the keys in both dictionaries
for key in aqa_scores_data.keys():
    # Check if the key exists in both dictionaries
    if key in class_data:
        # Create an array with classification and score
        combined_scores_data[key] = [class_data[key], aqa_scores_data[key]]

# Write the combined data to a new JSON file
with open(out_path, 'w') as file:
    json.dump(combined_scores_data, file, indent=2)

print(f"Combined scores data has been saved to: {out_path}")
