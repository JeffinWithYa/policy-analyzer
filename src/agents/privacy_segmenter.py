from typing import Any, AsyncGenerator
import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from agents.chatbot import AgentState, wrap_model
from agents.models import models
from service.logging_config import logger

SYSTEM_PROMPT = """You are a privacy policy segmentation expert. Your task is to break down privacy policies into individual clauses or segments that each represent a distinct privacy practice or policy statement.

Rules for segmentation:
1. Each segment should contain ONE complete privacy-related statement or practice
2. Preserve the original text exactly as written
3. Do not include headers, navigation elements, or non-privacy content
4. Keep segments to a reasonable length (roughly 1-3 sentences)
5. Ensure each segment can stand alone and be understood without context

Output format must be a JSON array of objects with numeric keys and segment values:
[{"0": "first segment text..."}, {"1": "second segment text..."}, ...]"""


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call the model and process its response"""
    m = models[config["configurable"].get("model", "gpt-4")]
    model_runnable = wrap_model(m)

    # Add system prompt to the messages
    messages = [HumanMessage(content=SYSTEM_PROMPT)] + state["messages"]
    modified_state = state.copy()
    modified_state["messages"] = messages

    logger.info("Sending messages to LLM:")
    for msg in state["messages"]:
        logger.info(f"Sent to LLM: {msg.content[:200]}...")  # Log first 200 chars

    accumulated_content = ""
    try:
        async for chunk in model_runnable.astream(modified_state, config):
            accumulated_content += chunk.content

        # Process the complete response
        try:
            content = accumulated_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            parsed_json = json.loads(content.strip())
            if not isinstance(parsed_json, list):
                logger.error("Invalid response format: not a JSON array")
                raise ValueError("Response must be a JSON array")

            # Validate format of each item
            for item in parsed_json:
                if not isinstance(item, dict) or len(item) != 1:
                    logger.error(f"Invalid segment format: {item}")
                    raise ValueError(
                        "Each item must be an object with exactly one numeric key"
                    )
                key = next(iter(item))
                if not isinstance(item[key], str):
                    logger.error(f"Invalid segment value type for key {key}")
                    raise ValueError("Segment values must be strings")

            logger.info(f"Successfully segmented into {len(parsed_json)} parts")
            logger.debug(f"Full response: {json.dumps(parsed_json)}")
            return {"messages": [AIMessage(content=json.dumps(parsed_json))]}

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error processing response: {str(e)}")
            error_response = [{"error": str(e)}]
            return {"messages": [AIMessage(content=json.dumps(error_response))]}

    except Exception as e:
        logger.error(f"Model error: {str(e)}")
        error_response = [{"error": f"Model error: {str(e)}"}]
        return {"messages": [AIMessage(content=json.dumps(error_response))]}


# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("model", acall_model)
workflow.set_entry_point("model")
workflow.add_edge("model", END)

# Export the compiled graph
privacy_segmenter = workflow.compile()
