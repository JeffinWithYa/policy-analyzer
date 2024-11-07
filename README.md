# Privacy Policy Analyzer

A tool for analyzing privacy policies using AI. This system processes privacy policy text segments and categorizes them according to standard privacy policy categories like data collection, sharing, retention, and user controls.

This repository is based off a blueprint and full toolkit for a LangGraph-based agent service architecture. It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface.
**[🎥 Watch a video walkthrough of the template repo and app](https://www.youtube.com/watch?v=VqQti9nGoe4)**

## Features

- Automated analysis of privacy policy segments
- Categorization into standard privacy categories:
  - First Party Collection/Use
  - Third Party Sharing/Collection
  - User Choice/Control
  - User Access, Edit, and Deletion
  - Data Retention
  - Data Security
  - Policy Change
  - Do Not Track
  - International and Specific Audiences
  - Other

- Structured output with explanations for each categorization
- Comparison with human annotations for accuracy assessment

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd policy-analyzer
```

2. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_key_here
```

3. Start the services with Docker Compose:
```bash
docker compose watch
```

## Running Analysis

4. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

5. The agent service API will be available at `http://localhost:80`. You can also use the OpenAPI docs at `http://localhost:80/redoc`.

6. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.


2. Run the analysis script:
```bash
docker exec policy-analyzer-agent_service-1 python agents/run_experiment.py
```

3. Check results in `data/analysis_results.json`:
```bash
cat data/analysis_results.json
```

The results will contain:
- Original policy segments
- Human annotations (if provided)
- AI model analysis with categories and explanations
- Match status between human and AI annotations

## Project Structure

```
.
├── data/                  # Data files
│   ├── records_test.json  # Example input data
│   └── analysis_results.json  # Generated results
├── src/                   # Source code
│   ├── agents/           # AI agent implementations
│   │   ├── privacy_policy_analyzer.py  # Main analyzer logic
│   │   └── run_experiment.py   # Analysis script
│   ├── service/          # API service
│   ├── schema/           # Data models
│   └── client/           # API client
├── docker/               # Docker configuration
│   ├── Dockerfile.app    # Streamlit app container
│   └── Dockerfile.service # Agent service container
└── compose.yaml         # Docker Compose configuration
```

## Development

The project uses Docker Compose Watch for development. Changes to source files will automatically trigger container rebuilds.

### Adding New Categories

To add new privacy policy categories:
1. Update `PRIVACY_LABELS` in `src/agents/privacy_policy_analyzer.py`
2. Update the system prompt to include the new categories
3. Rebuild the containers: `docker compose build`

### Testing

Use the provided `records_test.json` to verify your setup and test new features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Analysis Output Example

The `analysis_results.json` contains detailed results for each policy segment. Here's an example:

```json
{
    "document": 3828,
    "policyURL": "http://www.kraftrecipes.com/about/privacynotice.aspx",
    "segment": "Effective Date: May 7, 2015 Kraft Site Privacy Notice...",
    "human_annotations": [
      {
        "Other": {
          "Other Type": "Introductory/Generic"
        }
      },
      {
        "Policy Change": {
          "Change Type": "Unspecified",
          "User Choice": "Unspecified",
          "Notification Type": "General notice in privacy policy"
        }
      }
    ],
    "model_analysis": {
      "category": {
        "Other": {
          "Other Type": "Introductory/Generic"
        }
      },
      "explanation": "This segment serves as an introductory statement..."
    },
    "matching_details": {
      "top_level_match": true,
      "exact_match": true,
      "matching_subcategories": 1,
      "total_subcategories": 1,
      "subcategory_match_ratio": 1.0,
      "matched_categories": [
        {
          "Other": {
            "Other Type": "Introductory/Generic"
          }
        }
      ]
    }
}
```

### Understanding Match Results

The matching_details field provides metrics on how well the model's categorization matched human annotations:

- `top_level_match`: True if the main category (e.g., "Other") matches at least one human annotation
- `exact_match`: True if there's a perfect match between the model's category and one of the human annotations (including all subcategories)
- `matching_subcategories`: Number of matching subcategories
- `total_subcategories`: Total number of subcategories to match
- `subcategory_match_ratio`: Ratio of matching subcategories (1.0 = perfect match)
- `matched_categories`: List of human annotation categories that matched

In this example, the model correctly identified the segment as "Other: Introductory/Generic", matching one of the human annotations perfectly. Note that human annotators may identify multiple applicable categories (in this case, both "Other" and "Policy Change"), while the model currently focuses on the primary category.
