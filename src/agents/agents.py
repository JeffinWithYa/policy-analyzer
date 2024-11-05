from langgraph.graph.state import CompiledStateGraph

from agents.chatbot import chatbot
from agents.research_assistant import research_assistant
from agents.privacy_policy_analyzer import privacy_analyzer

DEFAULT_AGENT = "research-assistant"


agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
    "privacy-analyzer": privacy_analyzer,
}
