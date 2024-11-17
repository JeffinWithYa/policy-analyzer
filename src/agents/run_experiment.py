import json
import asyncio
from typing import Any
from pathlib import Path
import httpx
import traceback
import time
from typing import Optional, Dict, Any
import re


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
        "matched_categories": [],
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


class ServiceConnectionManager:
    def __init__(
        self, base_url: str, max_retries: int = 3, initial_backoff: float = 1.0
    ):
        self.base_url = base_url
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.current_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        await self.create_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def create_client(self):
        if self.current_client:
            await self.close()

        self.current_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=2200.0,  # Increased to 20 minutes to handle long rate limit waits
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    async def close(self):
        if self.current_client:
            await self.current_client.aclose()
            self.current_client = None

    async def make_request(
        self, segment: str, model: str = "claude-3-haiku"
    ) -> Dict[str, Any]:
        backoff_time = self.initial_backoff

        for attempt in range(self.max_retries):
            try:
                if not self.current_client:
                    await self.create_client()

                response = await self.current_client.post(
                    "/privacy-analyzer/invoke",
                    json={"message": segment, "model": model},
                )
                response.raise_for_status()
                return response.json()

            except (httpx.HTTPStatusError, httpx.HTTPError) as e:
                error_msg = str(e)
                # Look for rate limit information in both the error message and response
                wait_time_match = re.search(r"try again in (\d+)m([\d.]+)s", error_msg)

                if wait_time_match or "Rate limit" in error_msg:
                    if wait_time_match:
                        minutes, seconds = wait_time_match.groups()
                        wait_seconds = int(minutes) * 60 + float(seconds)
                    else:
                        # Default to a conservative wait if no specific time given
                        wait_seconds = 300  # 5 minutes

                    print(f"Rate limit reached. Waiting for {wait_seconds} seconds...")
                    await asyncio.sleep(wait_seconds)
                    # Create a new client after rate limit wait
                    await self.create_client()
                    continue

                # For any other error, use exponential backoff
                print(
                    f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    wait_time = backoff_time * (2**attempt)
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                print(
                    f"Connection error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )
                await self.close()

                if attempt < self.max_retries - 1:
                    wait_time = backoff_time * (2**attempt)
                    print(f"Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise


async def analyze_privacy_policies(input_file: str, output_file: str) -> None:
    """
    Analyze privacy policy segments and compare with annotations.
    """
    print(f"Loading data from {input_file}")
    try:
        with open(input_file, "r") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    results = []
    failure_count = 0
    last_save = time.time()
    save_interval = 300  # Save every 5 minutes

    async with ServiceConnectionManager("http://agent_service") as service:
        for i, record in enumerate(records):
            try:
                print(f"\nProcessing record {i+1}/{len(records)}")

                try:
                    analysis = await service.make_request(record["segment"])
                    failure_count = 0  # Reset counter on success

                    # Parse the response
                    try:
                        model_analysis = json.loads(analysis["content"])
                        model_category = model_analysis.get("category", {})
                        if isinstance(model_category, str):
                            try:
                                model_category = json.loads(model_category)
                            except json.JSONDecodeError:
                                model_category = {}

                        match_details = compare_categories(
                            model_category, record["annotationCategories"]
                        )

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing model response: {e}")
                        print(f"Raw response content: {analysis['content']}")
                        model_analysis = analysis["content"]
                        match_details = {
                            "top_level_match": False,
                            "exact_match": False,
                            "matching_subcategories": 0,
                            "total_subcategories": 0,
                            "matched_categories": [],
                        }

                except Exception as e:
                    failure_count += 1
                    print(f"Request failed (failure #{failure_count}): {str(e)}")

                    if failure_count >= 5:
                        print(
                            "Too many consecutive failures. Saving progress and exiting..."
                        )
                        break

                    continue

                # Create result entry
                result = {
                    "document": record["document"],
                    "policyURL": record["policyURL"],
                    "segment": record["segment"],
                    "human_annotations": record["annotationCategories"],
                    "model_analysis": model_analysis,
                    "matching_details": match_details,
                }

                results.append(result)

                # Save periodically or after failures
                current_time = time.time()
                if current_time - last_save > save_interval or failure_count > 0:
                    try:
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=2)
                        print(f"Saved {len(results)} results to file")
                        last_save = current_time
                    except Exception as e:
                        print(f"Error saving intermediate results: {e}")

            except Exception as e:
                print(
                    f"Error processing record from document {record['document']}: {e}"
                )
                print(f"Full error details: {traceback.format_exc()}")
                continue

    # Save final results
    print(f"\nProcessed {len(results)} records successfully")
    print(f"Writing final results to {output_file}")

    try:
        with open(output_file, "w") as f:
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
