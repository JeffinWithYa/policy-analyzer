import json
import asyncio
from typing import Any
from pathlib import Path
import httpx
import traceback

def compare_categories(model_category: dict, human_categories: list) -> dict:
    """
    Compare model predictions with human annotations in detail.
    Returns dict with matching statistics.
    """
    results = {
        "top_level_match": False,
        "exact_match": False,
        "matching_subcategories": 0,
        "total_subcategories": 0,
        "matched_categories": []
    }
    
    if not model_category:
        return results
    
    model_top_level = next(iter(model_category))  # Get the top-level category name
    model_subcategories = model_category[model_top_level]
    
    for human_category in human_categories:
        human_top_level = next(iter(human_category))  # Get the top-level category name
        
        # Check top-level match
        if model_top_level == human_top_level:
            results["top_level_match"] = True
            human_subcategories = human_category[human_top_level]
            
            # Count matching subcategories
            matching_sub = 0
            total_sub = len(human_subcategories)
            results["total_subcategories"] += total_sub
            
            for key, value in human_subcategories.items():
                if key in model_subcategories and model_subcategories[key] == value:
                    matching_sub += 1
            
            results["matching_subcategories"] += matching_sub
            
            # Check for exact match (all subcategories match)
            if human_subcategories == model_subcategories:
                results["exact_match"] = True
                results["matched_categories"].append(human_category)

    return results

async def analyze_privacy_policies(input_file: str, output_file: str) -> None:
    """
    Analyze privacy policy segments and compare with annotations.
    """
    async with httpx.AsyncClient(base_url="http://agent_service", timeout=30.0) as client:
        print(f"Loading data from {input_file}")
        try:
            with open(input_file, 'r') as f:
                records = json.load(f)
            print(f"Loaded {len(records)} records")
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
        
        results = []
        
        for i, record in enumerate(records):
            try:
                print(f"\nProcessing record {i+1}/{len(records)}")
                
                # Send request to the privacy analyzer
                response = await client.post(
                    "/privacy-analyzer/invoke",
                    json={
                        "message": record["segment"],
                        "model": "gpt-4"
                    }
                )
                response.raise_for_status()
                
                # Parse the response
                analysis = response.json()
                
                try:
                    # Parse the content string into a Python dict
                    model_analysis = json.loads(analysis["content"])
                    
                    # Debug print to see model response structure
                    print("Model response:", model_analysis)
                    
                    # Get the category from model response - adjust based on actual response structure
                    model_category = model_analysis.get("category", {})
                    if isinstance(model_category, str):
                        # If category is a string, try to parse it as JSON
                        try:
                            model_category = json.loads(model_category)
                        except json.JSONDecodeError:
                            model_category = {}
                    
                    # Get detailed matching statistics
                    match_details = compare_categories(model_category, record["annotationCategories"])
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing model response: {e}")
                    print(f"Raw response content: {analysis['content']}")
                    model_analysis = analysis["content"]
                    match_details = {
                        "top_level_match": False,
                        "exact_match": False,
                        "matching_subcategories": 0,
                        "total_subcategories": 0,
                        "matched_categories": []
                    }
                
                # Create result entry
                result = {
                    "document": record["document"],
                    "policyURL": record["policyURL"],
                    "segment": record["segment"],
                    "human_annotations": record["annotationCategories"],
                    "model_analysis": model_analysis,
                    "matching_details": {
                        "top_level_match": match_details["top_level_match"],
                        "exact_match": match_details["exact_match"],
                        "matching_subcategories": match_details["matching_subcategories"],
                        "total_subcategories": match_details["total_subcategories"],
                        "subcategory_match_ratio": (
                            match_details["matching_subcategories"] / match_details["total_subcategories"]
                            if match_details["total_subcategories"] > 0 else 0
                        ),
                        "matched_categories": match_details["matched_categories"]
                    }
                }
                
                results.append(result)
                print(f"Processed segment from document {record['document']}")
                print(f"Matching details: Top-level match: {match_details['top_level_match']}, "
                      f"Exact match: {match_details['exact_match']}, "
                      f"Matching subcategories: {match_details['matching_subcategories']}/{match_details['total_subcategories']}")
                
            except Exception as e:
                print(f"Error processing record from document {record['document']}: {e}")
                print(f"Full error details: {traceback.format_exc()}")
                continue
            
            # Save results periodically
            if len(results) % 10 == 0:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Saved {len(results)} results to file")
                except Exception as e:
                    print(f"Error saving intermediate results: {e}")
        
        print(f"\nProcessed {len(results)} records successfully")
        print(f"Writing final results to {output_file}")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print("Results written successfully")
        except Exception as e:
            print(f"Error writing output file: {e}")

if __name__ == "__main__":
    INPUT_FILE = "/app/data/records_consolidated.json"
    OUTPUT_FILE = "/app/data/analysis_results.json"
    
    print("Starting privacy policy analysis...")
    asyncio.run(analyze_privacy_policies(INPUT_FILE, OUTPUT_FILE))
    print("Done!")