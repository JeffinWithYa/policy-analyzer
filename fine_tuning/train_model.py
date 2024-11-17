from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open('C:/Users/MCTI Student/Desktop/privacy_proj/policy-analyzer/data/records_consolidated.json', "rb"),
  purpose="fine-tune"
)

"""
from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.create(
  training_file="file-abc123",
  model="gpt-4o-mini-2024-07-18"
)
"""
