from typing import Annotated, Sequence, TypedDict, List, Dict
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


def create_gdpr_agent():
    """Create the GDPR compliance analysis agent"""

    # Store components at module level
    retriever_tool = None
    segment_lookup = None
    vectorstore = None

    def setup_retrieval(privacy_segments: List[Dict], update_existing: bool = False):
        """Setup or update vectorstore and retriever"""
        nonlocal retriever_tool, segment_lookup, vectorstore

        if not privacy_segments:
            logger.debug(
                "No privacy segments provided - deferring vectorstore creation"
            )
            return None, None, None

        if retriever_tool is None or update_existing:
            logger.info(
                f"{'Updating' if update_existing else 'Setting up'} retrieval for {len(privacy_segments)} segments"
            )
            vectorstore, segment_lookup = create_or_update_vectorstore(
                privacy_segments,
                existing_vectorstore=vectorstore if update_existing else None,
                existing_lookup=segment_lookup if update_existing else None,
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3, "include_metadata": True},
            )
            retriever_tool = create_retriever_tool(
                retriever,
                "search_privacy_policy",
                "Search privacy policy segments for GDPR compliance information",
            )

        return retriever_tool, segment_lookup, vectorstore

    def agent(state):
        """Agent to decide whether to retrieve or generate answer"""
        nonlocal retriever_tool, segment_lookup, vectorstore

        messages = state["messages"]
        privacy_segments = state.get("privacy_segments", [])
        update_store = state.get("update_vectorstore", False)

        # Initialize or update retrieval components if we have segments
        if privacy_segments:
            retriever_tool, segment_lookup, vectorstore = setup_retrieval(
                privacy_segments, update_existing=update_store
            )

        if retriever_tool is None:
            logger.debug("No retriever tool available - returning empty response")
            return {
                **state,
                "messages": [
                    AIMessage(content="No privacy segments available for analysis")
                ],
            }

        model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        model = model.bind_tools([retriever_tool])
        response = model.invoke(messages)

        return {**state, "messages": [response], "segment_lookup": segment_lookup}

    def generate_answer(state):
        """Generate GDPR compliance answer from retrieved segments"""
        nonlocal segment_lookup

        messages = state["messages"]
        question = messages[0].content
        retrieved_docs = messages[-1].content

        logger.info(f"Generating answer for: {question}")

        # Parse retrieved documents and look up original segments
        retrieved_segments = []
        for doc in retrieved_docs.split("\n\n"):
            if doc.strip() and "Segment:" in doc:
                original_segment = segment_lookup.get(doc.strip())
                if original_segment:
                    retrieved_segments.append(
                        {
                            "segment": original_segment["segment"],
                            "model_analysis": original_segment["model_analysis"],
                        }
                    )

        logger.info(f"Found {len(retrieved_segments)} relevant segments")

        prompt = PromptTemplate(
            template="""Based on the retrieved privacy policy segments, answer the following GDPR compliance question:
            Question: {question}
            Retrieved segments: {context}
            
            Provide a list of relevant segments that address this question. If no relevant segments are found, state that explicitly.""",
            input_variables=["question", "context"],
        )

        model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        response = model.invoke(
            prompt.format(question=question, context=retrieved_docs)
        )

        return {
            **state,
            "messages": [response],
            "results": {question: retrieved_segments},
        }

    # Create graph
    workflow = StateGraph(GDPRState)

    # Add nodes
    workflow.add_node("agent", agent)
    retrieve = ToolNode([])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_answer)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        lambda x: (
            "retrieve"
            if "search_privacy_policy" in x["messages"][-1].content
            else "generate"
        ),
        {"retrieve": "retrieve", "generate": "generate"},
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def analyze_gdpr_compliance(privacy_segments: List[Dict]) -> Dict[str, List[Dict]]:
    """Run GDPR compliance analysis on privacy policy segments"""
    logger.info(f"Starting GDPR analysis with {len(privacy_segments)} segments")
    agent = create_gdpr_agent()
    results = {}

    for question in GDPR_QUESTIONS:
        logger.info(f"Processing question: {question}")
        inputs = {
            "messages": [
                HumanMessage(content=question)
            ],  # Use HumanMessage instead of tuple
            "results": {},
            "privacy_segments": privacy_segments,
        }

        try:
            output = agent.invoke(inputs)
            logger.info(f"Got response for question: {question}")
            logger.debug(f"Response details: {output}")
            results[question] = output["results"][question]
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}", exc_info=True)
            results[question] = []

    logger.info("GDPR analysis complete")
    return results
