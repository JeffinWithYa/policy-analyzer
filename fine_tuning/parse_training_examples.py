import json
from collections import defaultdict
import random

# Function to process data and balance classes with a total limit
def process_json_file_limited(input_file, output_file, classes, max_samples_per_class, total_samples):
    # Load the JSON data from the file
    with open(input_file, 'r') as file:
        input_data = json.load(file)
    
    # System content
    system_content = "You are a privacy policy analyzer. Your task is to analyze privacy policy segments and categorize them according to a specific schema."
    
    # Create a dictionary to store samples by class
    class_samples = defaultdict(list)
    
    # Group data by classes
    for entry in input_data:
        user_content = entry["segment"]
        first_annotation = entry["annotationCategories"][0]
        category = next(iter(first_annotation.keys()))  # Get the top-level class
        if category in classes:
            assistant_content = json.dumps(first_annotation, ensure_ascii=False)
            message = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            class_samples[category].append(message)
    
    # Balance the classes and limit the total number of samples
    balanced_data = []
    for class_name in classes:
        samples = class_samples[class_name]
        balanced_data.extend(samples[:max_samples_per_class])  # Limit samples per class
    
    # Shuffle the data to randomize order
    random.shuffle(balanced_data)
    
    # Limit the total number of samples
    limited_data = balanced_data[:total_samples]
    
    # Write the limited and balanced data to the output file
    with open(output_file, 'w') as file:
        for item in limited_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed limited and balanced data has been written to {output_file}")

# File paths
input_file = 'C:/Users/MCTI Student/Desktop/privacy_proj/policy-analyzer/data/records_consolidated.json' 
output_file = 'output_limited_balanced.json'  # Replace with the desired output file path

# Define the ten classes, max samples per class, and total samples
classes = [
    "Data Retention", "Data Security", "Do Not Track", "First Party Collection/Use",
    "International and Specific Audiences", "Other", "Policy Change",
    "Third Party Sharing/Collection", "User Access, Edit and Deletion", "User Choice/Control"
]
max_samples_per_class = 10  # Max samples per class
total_samples = 100  # Total number of samples in the output

# Process the input JSON and write the output
process_json_file_limited(input_file, output_file, classes, max_samples_per_class, total_samples)
