import json
from collections import defaultdict

def analyze_privacy_policy_categories(data, output_file):
    # Structure to store hierarchy
    hierarchy = defaultdict(lambda: defaultdict(set))
    
    # Go through each annotation
    for entry in data:
        annotation_category = entry.get('annotationCategory', {})
        
        # Each entry has one high-level category
        for high_level_cat, subcats in annotation_category.items():
            # Add all subcategories and their values
            for subcat, value in subcats.items():
                hierarchy[high_level_cat][subcat].add(value)
    
    # Write the hierarchy to a file in a readable format
    with open(output_file, 'w') as f:
        for high_level_cat in sorted(hierarchy.keys()):
            f.write(f"\n# {high_level_cat}\n")
            for subcat in sorted(hierarchy[high_level_cat].keys()):
                values = sorted(hierarchy[high_level_cat][subcat])
                f.write(f"\n## {subcat}\n")
                for value in values:
                    f.write(f"- {value}\n")

# Load and parse the JSON data
with open('records_sanitized.json', 'r') as f:
    data = json.load(f)

# Analyze and save to output file
analyze_privacy_policy_categories(data, 'privacy_categories_hierarchy.txt')