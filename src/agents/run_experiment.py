import json
import asyncio
from typing import Any
from pathlib import Path
import httpx

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
                        "model": "gpt-4o-mini"
                    }
                )
                response.raise_for_status()
                
                # Parse the response
                analysis = response.json()
                
                try:
                    # Parse the content string into a Python dict
                    model_analysis = json.loads(analysis["content"])
                    
                    # Compare categories for match
                    human_category = record["annotationCategory"]
                    model_category = model_analysis.get("category", {})
                    match = human_category == model_category
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing model response: {e}")
                    model_analysis = analysis["content"]
                    match = False
                
                # Create result entry
                result = {
                    "document": record["document"],
                    "annotationID": record["annotationID"],
                    "segment": record["segment"],
                    "human_annotation": record["annotationCategory"],
                    "model_analysis": model_analysis,
                    "match": match
                }
                
                results.append(result)
                print(f"Processed annotation {record['annotationID']} from document {record['document']}")
                
            except Exception as e:
                print(f"Error processing record {record['annotationID']}: {e}")
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
    INPUT_FILE = "/app/data/records_test.json"
    OUTPUT_FILE = "/app/data/analysis_results.json"
    
    print("Starting privacy policy analysis...")
    asyncio.run(analyze_privacy_policies(INPUT_FILE, OUTPUT_FILE))
    print("Done!")