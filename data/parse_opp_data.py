import csv
import json
import os
import re
import html
from pathlib import Path
from typing import Dict, List


class Policy:
    def __init__(self, segments: List[str]):
        self.segments = segments


class PolicyRecord:
    def __init__(self, doc_id: int, annotation_id: int, policy_url: str, segment: str, annotation_category: Dict[str, Dict[str, str]]):
        self.doc_id = doc_id
        self.annotation_id = annotation_id
        self.policy_url = policy_url
        self.segment = segment
        self.annotation_category = annotation_category

    def to_dict(self):
        return {
            "document": self.doc_id,
            "annotationID": self.annotation_id,
            "policyURL": self.policy_url,
            "segment": self.segment,
            "annotationCategory": self.annotation_category
        }


def sanitize_html(content: str, policy: str) -> str:
    if policy == "strict":
        # Remove all HTML tags and unescape HTML entities
        return html.unescape(re.sub(r"<.*?>", "", content))
    elif policy == "ugc":
        # Remove dangerous tags for user-generated content
        safe_tags = ["b", "i", "u", "em", "strong", "a"]
        return html.unescape(re.sub(r"<(?!\/?(" + "|".join(safe_tags) + r")\b)[^>]*>", "", content))
    else:
        return content


def read_policy(filename: str) -> Policy:
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        segments = content.split("|||")
        return Policy(segments)


def read_policies_from_dir(directory: str) -> Dict[str, Policy]:
    policies = {}
    for file_path in Path(directory).glob("*"):
        if file_path.is_file():
            try:
                policy_name = file_path.stem
                policies[policy_name] = read_policy(str(file_path))
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return policies


def get_policy_segment(policies: Dict[str, Policy], policy_name: str, segment_idx: int) -> str:
    if policy_name not in policies:
        raise ValueError(f"Policy with the name '{policy_name}' not found")
    policy = policies[policy_name]
    if segment_idx < 0 or segment_idx >= len(policy.segments):
        raise IndexError(f"Segment index {segment_idx} out of bounds for policy {policy_name}")
    return policy.segments[segment_idx]


def parse_annotation_category(high_level_category: str, json_string: str) -> Dict[str, Dict[str, str]]:
    try:
        category_data = json.loads(json_string)
        annotation_category = {
            high_level_category: {
                key: value.get("value", "Unspecified") for key, value in category_data.items()
            }
        }
        return annotation_category
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def read_annotations(filename: str, sanitize_option: str, policies: Dict[str, Policy]) -> List[PolicyRecord]:
    records = []
    annotation_id = 1

    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for record in reader:
            try:
                doc_id = int(record[3])
                policy_name = Path(filename).stem
                segment_idx = int(record[4])
                segment = get_policy_segment(policies, policy_name, segment_idx)
                sanitized_segment = sanitize_html(segment, sanitize_option)

                # Get high-level category from column 5
                high_level_category = record[5]

                # Parse the JSON from column 6 and include it under the high-level category
                annotation_category = parse_annotation_category(high_level_category, record[6])

                policy_record = PolicyRecord(
                    doc_id, annotation_id, record[8], sanitized_segment, annotation_category
                )
                records.append(policy_record)
                annotation_id += 1
            except (ValueError, IndexError) as e:
                print(f"Error processing record {record}: {e}")
    return records


def read_annotations_from_dir(directory: str, sanitize_option: str, policies: Dict[str, Policy]) -> List[PolicyRecord]:
    all_records = []
    for file_path in Path(directory).glob("*"):
        if file_path.is_file():
            try:
                records = read_annotations(str(file_path), sanitize_option, policies)
                all_records.extend(records)
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
    return all_records


def write_records_to_json(records: List[PolicyRecord], out_file: str):
    try:
        with open(out_file, "w", encoding="utf-8") as json_file:
            json_data = [record.to_dict() for record in records]
            json.dump(json_data, json_file, indent=2)
        print(f"JSON data successfully written to {out_file}")
    except Exception as e:
        print(f"Error writing to {out_file}: {e}")


def main():
    policies_dir = "sanitized_policies"
    annotations_dir = "consolidation"
    out_files = {
        "strict": "output/records_sanitized.json",
        "ugc": "output/records_ugc_sanitized.json",
        "": "output/records_not_sanitized.json"
    }

    policies = read_policies_from_dir(policies_dir)

    for sanitize_option, out_file in out_files.items():
        records = read_annotations_from_dir(annotations_dir, sanitize_option, policies)
        write_records_to_json(records, out_file)


if __name__ == "__main__":
    main()
