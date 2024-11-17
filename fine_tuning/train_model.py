from openai import OpenAI
import time

client = OpenAI()

# Upload the fine-tuning file
"""
client.files.create(
    file=open(
        "/Users/jeffreyjeyachandren/Desktop/policy-analyzer/policy-analyzer/fine_tuning/output_limited_balanced.jsonl",
        "rb",
    ),
    purpose="fine-tune",
)
"""

"""
# List all files
files = client.files.list()

# Print each file's id and name
for file in files:
    print(f"File ID: {file.id}, Filename: {file.filename}, Purpose: {file.purpose}")
"""

"""
# Fine-tune the model
client.fine_tuning.jobs.create(
    training_file="file-yTOJ45lteVtUB1NAf8grqOL9", model="gpt-4o-mini-2024-07-18"
)
"""
"""
# List all fine-tuning jobs
jobs = client.fine_tuning.jobs.list()

# Print details of each job
for job in jobs:
    print(f"Job ID: {job.id}")
    print(f"Model: {job.model}")
    print(f"Status: {job.status}")
    print(f"Created at: {job.created_at}")
    print("---")
"""

# Continuously monitor the fine-tuning job status

while True:
    job_status = client.fine_tuning.jobs.retrieve("ftjob-VJowv6ZvPPUgX89LNpiAqTgf")
    print(f"Status: {job_status.status}")

    if job_status.status in ["succeeded", "failed", "cancelled"]:
        break

    time.sleep(60)  # Wait 60 seconds before checking again