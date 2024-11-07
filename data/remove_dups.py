import json

def consolidate_records(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store consolidated records
    consolidated = {}
    
    # Process each record
    for record in data:
        segment = record['segment']
        
        if segment not in consolidated:
            # Create new consolidated record
            consolidated[segment] = {
                'document': record['document'],
                'policyURL': record['policyURL'],
                'segment': segment,
                'annotationCategories': []
            }
        
        # Add annotation if it's not already present
        if record['annotationCategory'] not in consolidated[segment]['annotationCategories']:
            consolidated[segment]['annotationCategories'].append(record['annotationCategory'])
    
    # Convert dictionary to list
    result = list(consolidated.values())
    
    # Write the consolidated data to output file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return len(data), len(result)

# Usage
input_file = 'records_sanitized.json'
output_file = 'records_consolidated.json'
original_count, new_count = consolidate_records(input_file, output_file)
print(f"Consolidated {original_count} records into {new_count} unique records")