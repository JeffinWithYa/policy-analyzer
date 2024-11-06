from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.models import models

class AgentState(MessagesState, total=False):
    """State for privacy policy analyzer"""

SYSTEM_PROMPT = """You are a privacy policy analyzer. Your task is to analyze privacy policy segments and categorize them according to a specific schema.

When analyzing a privacy policy segment, you should output your analysis in the following JSON format:

{
    "category": {
        "Primary Category": {
            "Sub Category": "specific type"
        }
    },
    "explanation": "Brief explanation of why this categorization was chosen"
}

The main categories and their sub-categories are:
1. First Party Collection/Use
   - Collection Mode
   - Information Type
   - Purpose
   - User Choice/Control
   
2. Third Party Sharing/Collection
   - Third Party Entity
   - Purpose
   - Information Type
   - User Choice/Control

3. User Access, Edit, & Deletion
   - Access Type
   - User Choice/Control

4. Data Security
   - Security Measure Type

5. Data Retention
   - Retention Period
   - Retention Purpose

6. Policy Change
   - Change Type
   - User Choice
   - Notification Type

7. Other
   - Other Type (e.g., "Introductory/Generic", "Contact Information", etc.)

Always output your analysis in the exact JSON format specified above. Be concise but precise in your explanations."""

def wrap_model(model):
    """Wrap the model with the system prompt"""
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call the model and process its response"""
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    return {"messages": [response]}

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

privacy_analyzer = agent.compile(
    checkpointer=MemorySaver(),
)
