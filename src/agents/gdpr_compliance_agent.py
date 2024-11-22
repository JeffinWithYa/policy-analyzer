from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
import logging
import json

# Configure logger
logger = logging.getLogger(__name__)

GDPR_QUESTIONS = [
    "What are the identity and contact details of the organization?",
    "What are the purposes and legal basis for processing personal data?",
    "What are the legitimate interests of the organization?",
    "Who are the recipients of personal data?",
    "Are there any international data transfers and what safeguards exist?",
    "What is the data retention period?",
    "What are the data subject rights?",
    "How can consent be withdrawn?",
    "How can complaints be lodged?",
    "What are the requirements for providing personal data?",
    "Is there automated decision-making or profiling?",
]


class GDPRState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    results: Dict[str, List[str]]


def create_vectorstore(privacy_segments: List[Dict]):
    """Create and populate vector store with privacy policy segments"""
    logger.info(f"Creating vectorstore with {len(privacy_segments)} segments")
    if not privacy_segments:
        logger.warning("No privacy segments provided, using dummy text")
        dummy_text = "No privacy policy segments available"
        vectorstore = Chroma.from_texts(
            texts=[dummy_text],
            metadatas=[{"original_segment": dummy_text}],
            embedding=OpenAIEmbeddings(),
            collection_name="privacy-policy-segments",
        )
        return vectorstore, {dummy_text: {"segment": dummy_text, "model_analysis": {}}}

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )

    texts = []
    metadatas = []
    segment_lookup = {}

    for i, segment in enumerate(privacy_segments):
        # Combine segment text with its annotation explanation for better context
        combined_text = f"""
        Segment: {segment['segment']}
        Category: {segment['model_analysis']['category']}
        Explanation: {segment['model_analysis']['explanation']}
        """
        texts.append(combined_text)
        metadatas.append({"original_segment": combined_text})
        segment_lookup[combined_text] = segment

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=OpenAIEmbeddings(),
        collection_name="privacy-policy-segments",
    )

    return vectorstore, segment_lookup


def analyze_gdpr_compliance(
    privacy_segments: List[Dict], update_vectorstore: bool = False
) -> Dict[str, List[Dict]]:
    """Run GDPR compliance analysis on privacy policy segments"""
    logger.info(f"Starting GDPR analysis with {len(privacy_segments)} segments")

    try:
        agent = create_gdpr_agent()
        logger.info("Created GDPR agent")
        results = {}

        for question in GDPR_QUESTIONS:
            logger.info(f"Processing question: {question}")
            inputs = {
                "messages": [HumanMessage(content=question)],
                "results": {},
                "privacy_segments": privacy_segments,
                "update_vectorstore": update_vectorstore,
            }

            try:
                logger.info("Invoking agent")
                output = agent.invoke(inputs)
                logger.info("Agent invocation complete")
                logger.debug(f"Agent output: {output}")

                results[question] = output.get("results", {}).get(question, [])
                logger.info(
                    f"Processed results for question: {len(results[question])} segments found"
                )

            except Exception as e:
                logger.error(
                    f"Error processing question '{question}': {e}", exc_info=True
                )
                results[question] = []

        logger.info("GDPR analysis complete")
        return results

    except Exception as e:
        logger.error(f"Error in analyze_gdpr_compliance: {e}", exc_info=True)
        raise


# Define state type
class AgentState(TypedDict):
    """Type for agent state"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_gdpr_agent():
    """Create the GDPR compliance analysis agent for use with /invoke endpoint"""
    logger.info("Creating GDPR agent")

    def grade_relevance(docs, question):
        """Grade if retrieved documents are relevant to the question"""
        # Map questions to key terms for relevance checking
        question_terms = {
            "What are the identity and contact details of the organization?": [
                "contact",
                "email",
                "address",
                "organization",
                "company",
                "identity",
            ],
            "What are the purposes and legal basis for processing personal data?": [
                "legal basis",
                "consent",
                "legitimate interest",
                "purpose",
                "processing",
            ],
            "What are the legitimate interests of the organization?": [
                "legitimate interest",
                "business purpose",
                "processing reason",
            ],
            "Who are the recipients of personal data?": [
                "recipient",
                "third party",
                "share",
                "transfer",
                "disclosure",
            ],
            "Are there any international data transfers and what safeguards exist?": [
                "international",
                "transfer",
                "safeguard",
                "overseas",
                "cross-border",
            ],
            "What is the data retention period?": [
                "retention",
                "store",
                "keep",
                "period",
                "duration",
            ],
            "What are the data subject rights?": [
                "right",
                "access",
                "rectification",
                "erasure",
                "portability",
            ],
            "How can consent be withdrawn?": [
                "withdraw",
                "consent",
                "opt-out",
                "revoke",
            ],
            "How can complaints be lodged?": [
                "complaint",
                "supervisory",
                "authority",
                "lodge",
            ],
            "What are the requirements for providing personal data?": [
                "requirement",
                "mandatory",
                "voluntary",
                "provide",
            ],
            "Is there automated decision-making or profiling?": [
                "automated",
                "profiling",
                "decision",
                "automatic",
            ],
        }

        terms = question_terms.get(question, [])
        for doc in docs:
            text = doc.page_content.lower()
            if any(term.lower() in text for term in terms):
                return True
        return False

    def process_segments(state):
        """Process privacy segments and create vectorstore"""
        messages = state["messages"]
        try:
            last_message = messages[-1].content
            data = json.loads(last_message)
            privacy_segments = data.get("privacy_segments", [])

            if not privacy_segments:
                return {
                    **state,
                    "messages": [
                        AIMessage(content="No privacy segments provided for analysis")
                    ],
                }

            vectorstore, segment_lookup = create_vectorstore(privacy_segments)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            results = {}
            for question in GDPR_QUESTIONS:
                docs = retriever.invoke(question)

                # Only include relevant documents
                if grade_relevance(docs, question):
                    relevant_segments = []
                    for doc in docs:
                        for key in segment_lookup:
                            if (
                                doc.page_content.strip() in key
                                or key in doc.page_content.strip()
                            ):
                                segment = segment_lookup[key]
                                if segment not in relevant_segments:
                                    relevant_segments.append(segment)
                    results[question] = relevant_segments
                else:
                    results[question] = []

            # Format response
            response = "GDPR Compliance Analysis Results:\n\n"
            for question, segments in results.items():
                response += f"Question: {question}\n"
                if segments:
                    for segment in segments:
                        response += f"- Segment: {segment['segment']}\n"
                        response += (
                            f"  Category: {segment['model_analysis']['category']}\n"
                        )
                        response += f"  Explanation: {segment['model_analysis']['explanation']}\n"
                else:
                    response += "No relevant segments found.\n"
                response += "\n"

            return {**state, "messages": [AIMessage(content=response)]}

        except Exception as e:
            logger.error(f"Error in GDPR analysis: {e}", exc_info=True)
            return {
                **state,
                "messages": [
                    AIMessage(
                        content=f"An error occurred during GDPR analysis: {str(e)}"
                    )
                ],
            }

    # Create simple workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("process", process_segments)
    workflow.add_edge(START, "process")
    workflow.add_edge("process", END)

    return workflow.compile()
