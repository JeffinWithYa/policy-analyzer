import json
import os
import re

def process_policy_data(input_file):
    # Create output directory if it doesn't exist
    output_dir = "processed_policies"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by policyURL
    policies = {}
    for item in data:
        url = item['policyURL']
        if url not in policies:
            policies[url] = []
            
        # Keep only segment and model_analysis
        simplified_item = {
            'segment': item['segment'],
            'model_analysis': item['model_analysis']
        }
        policies[url].append(simplified_item)
    
    # Write individual files
    for url, items in policies.items():
        # Create a safe filename from the URL
        filename = url.replace('http://', '').replace('https://', '')
        # Remove invalid filename characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit filename length
        if len(filename) > 200:  # Windows max path is 260, leaving room for directory and extension
            filename = filename[:200]
        filename = f"{output_dir}/{filename}.json"
        
        with open(filename, 'w') as f:
            json.dump(items, f, indent=2)
        print(f"Created {filename}")

# Use the function
process_policy_data(r'C:\Users\MCTI Student\Desktop\privacy_proj\policy-analyzer\data\analysis_results_gpt-4o-2024-08-06.json')
