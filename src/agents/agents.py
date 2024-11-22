from langgraph.graph.state import CompiledStateGraph

from agents.chatbot import chatbot
from agents.research_assistant import research_assistant
from agents.privacy_policy_analyzer import privacy_analyzer
from agents.privacy_segmenter import privacy_segmenter
from agents.gdpr_compliance_agent import create_gdpr_agent

DEFAULT_AGENT = "research-assistant"

agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
    "privacy-analyzer": privacy_analyzer,
    "privacy-segmenter": privacy_segmenter,
    "gdpr-compliance": create_gdpr_agent(),
}


def get_gdpr_agent():
    """Lazy initialization of GDPR agent"""
    global agents
    if agents["gdpr-compliance"] is None:
        agents["gdpr-compliance"] = create_gdpr_agent()
    return agents["gdpr-compliance"]
