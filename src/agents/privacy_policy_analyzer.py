from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from agents.models import models

import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from tqdm import tqdm

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

1. Data Retention
   - Personal Information Type: [Computer information, Contact, Cookies and tracking elements, Demographic, Financial, Generic personal information, Health, IP address and device IDs, Location, Other, Personal identifier, Social media data, Survey data, Unspecified, User online activities, User profile]
   - Retention Period: [Indefinitely, Limited, Other, Stated Period, Unspecified]
   - Retention Purpose: [Advertising, Analytics/Research, Legal requirement, Marketing, Other, Perform service, Service operation and security, Unspecified]

2. Data Security
   - Security Measure: [Data access limitation, Generic, Other, Privacy review/audit, Privacy training, Privacy/Security program, Secure data storage, Secure data transfer, Secure user authentication, Unspecified]

3. Do Not Track
   - Do Not Track policy: [Honored, Mentioned but unclear if honored, Not honored, Other]

4. First Party Collection/Use
   - Action First-Party: [Collect from user on other websites, Collect in mobile app, Collect on mobile website, Collect on website, Other, Receive from other parts of company/affiliates, Receive from other service/third-party (named), Receive from other service/third-party (unnamed), Track user on other websites, Unspecified]
   - Choice Scope: [Both, Collection, Use, Unspecified, not-selected]
   - Choice Type: [Browser/device privacy controls, Dont use service/feature, First-party privacy controls, Opt-in, Opt-out link, Opt-out via contacting company, Other, Third-party privacy controls, Unspecified, not-selected]
   - Collection Mode: [Explicit, Implicit, Unspecified, not-selected]
   - Does/Does Not: [Does, Does Not]
   - Identifiability: [Aggregated or anonymized, Identifiable, Other, Unspecified, not-selected]
   - Personal Information Type: [Computer information, Contact, Cookies and tracking elements, Demographic, Financial, Generic personal information, Health, IP address and device IDs, Location, Other, Personal identifier, Social media data, Survey data, Unspecified, User online activities, User profile]
   - Purpose: [Additional service/feature, Advertising, Analytics/Research, Basic service/feature, Legal requirement, Marketing, Merger/Acquisition, Other, Personalization/Customization, Service Operation and Security, Unspecified]
   - User Type: [Other, Unspecified, User with account, User without account, not-selected]

5. International and Specific Audiences
   - Audience Type: [Californians, Children, Citizens from other countries, Europeans, Other]

6. Other
   - Other Type: [Introductory/Generic, Other, Practice not covered, Privacy contact information]

7. Policy Change
   - Change Type: [In case of merger or acquisition, Non-privacy relevant change, Other, Privacy relevant change, Unspecified]
   - Notification Type: [General notice in privacy policy, General notice on website, No notification, Other, Personal notice, Unspecified]
   - User Choice: [None, Opt-in, Opt-out, Other, Unspecified, User participation]

8. Third Party Sharing/Collection
   - Action Third Party: [Collect on first party website/app, Other, Receive/Shared with, See, Track on first party website/app, Unspecified]
   - Choice Scope: [Both, Collection, Use, Unspecified, not-selected]
   - Choice Type: [Browser/device privacy controls, Dont use service/feature, First-party privacy controls, Opt-in, Opt-out link, Opt-out via contacting company, Other, Third-party privacy controls, Unspecified, not-selected]
   - Does/Does Not: [Does, Does Not]
   - Identifiability: [Aggregated or anonymized, Identifiable, Other, Unspecified, not-selected]
   - Personal Information Type: [Computer information, Contact, Cookies and tracking elements, Demographic, Financial, Generic personal information, Health, IP address and device IDs, Location, Other, Personal identifier, Survey data, Unspecified, User Profile, User online activities]
   - Purpose: [Additional service/feature, Advertising, Analytics/Research, Basic service/feature, Legal requirement, Marketing, Merger/Acquisition, Other, Personalization/Customization, Service operation and security, Unspecified]
   - Third Party Entity: [Named third party, Other, Other part of company/affiliate, Other users, Public, Unnamed third party, Unspecified]
   - User Type: [Other, Unspecified, User with account, User without account, not-selected]

9. User Access, Edit and Deletion
   - Access Scope: [Other, Other data about user, Profile data, Transactional data, Unspecified, User account data]
   - Access Type: [Deactivate account, Delete account (full), Delete account (partial), Edit information, Export, None, Other, Unspecified, View]
   - User Type: [Other, Unspecified, User with account, User without account, not-selected]

10. User Choice/Control
    - Choice Scope: [First party collection, First party use, Third party sharing/collection, Third party use, Unspecified]
    - Choice Type: [Browser/device privacy controls, Dont use service/feature, First-party privacy controls, Opt-in, Opt-out link, Opt-out via contacting company, Other, Third-party privacy controls, Unspecified]
    - Personal Information Type: [Computer information, Contact, Cookies and tracking elements, Demographic, Financial, Generic personal information, Health, IP address and device IDs, Location, Other, Personal identifier, Social media data, Survey data, Unspecified, User online activities, User profile]
    - Purpose: [Additional service/feature, Advertising, Analytics/Research, Basic service/feature, Legal requirement, Marketing, Merger/Acquisition, Other, Personalization/Customization, Service Operation and Security, Unspecified]
    - User Type: [Other, Unspecified, User with account, User without account, not-selected]

Always output your analysis in the exact JSON format specified above. Be concise but precise in your explanations."""

def wrap_model(model):
    """Wrap the model with the system prompt"""
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
async def process_record(record):
    try:
        # Your existing processing code here
        response = await openai.chat.completions.create(...)
        
    except Exception as e:
        logger.error(f"Error processing record: {str(e)}")
        # Add a small delay before retrying
        time.sleep(1)
        raise
        
async def process_records(records):
    with tqdm(total=len(records)) as pbar:
        for i, record in enumerate(records):
            try:
                result = await process_record(record)
                pbar.update(1)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to process record {i} after all retries: {str(e)}")
                pbar.update(1)
                continue

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
