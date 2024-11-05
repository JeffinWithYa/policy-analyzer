from datetime import datetime
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.models import models

PRIVACY_LABELS = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit, and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other"
]

class AgentState(MessagesState, total=False):
    """State for privacy policy analyzer"""

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are a privacy policy analyzer. Your task is to analyze privacy policy clauses and assign appropriate labels.

When given a privacy policy clause, you should:
1. Analyze the content carefully
2. Assign one or more of the following labels that best describe the clause:
{', '.join(PRIVACY_LABELS)}

Format your response as follows:
Labels: [list of applicable labels]
Explanation: [brief explanation of why these labels apply]

Today's date is {current_date}.
"""

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    
    # Save the analysis to a file
    with open("privacy_policy_analysis.txt", "a") as f:
        f.write(f"\n--- Analysis {datetime.now().isoformat()} ---\n")
        f.write(f"Clause: {state['messages'][-1].content}\n")
        f.write(f"Analysis: {response.content}\n")
        f.write("-" * 80 + "\n")
    
    return {"messages": [response]}

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

privacy_analyzer = agent.compile(
    checkpointer=MemorySaver(),
) 