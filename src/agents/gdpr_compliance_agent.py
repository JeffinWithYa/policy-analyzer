from typing import Annotated, Sequence, TypedDict, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages

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
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )

    documents = []
    segment_lookup = {}

    for i, segment in enumerate(privacy_segments):
        # Combine segment text with its annotation explanation for better context
        combined_text = f"""
        Segment: {segment['segment']}
        Category: {segment['model_analysis']['category']}
        Explanation: {segment['model_analysis']['explanation']}
        """
        documents.append(combined_text)
        segment_lookup[combined_text] = segment

    doc_splits = text_splitter.create_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        collection_name="privacy-policy-segments",
        metadatas=[{"original_segment": doc} for doc in documents],
    )

    return vectorstore, segment_lookup


def create_gdpr_agent(privacy_segments: List[Dict]):
    """Create the GDPR compliance analysis agent"""

    # Create vector store and retriever
    vectorstore, segment_lookup = create_vectorstore(privacy_segments)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3, "include_metadata": True}
    )

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "search_privacy_policy",
        "Search privacy policy segments for GDPR compliance information",
    )

    # Define agent nodes
    def agent(state):
        """Agent to decide whether to retrieve or generate answer"""
        messages = state["messages"]
        model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        model = model.bind_tools([retriever_tool])
        response = model.invoke(messages)
        return {"messages": [response]}

    def generate_answer(state):
        """Generate GDPR compliance answer from retrieved segments"""
        messages = state["messages"]
        question = messages[0].content
        retrieved_docs = messages[-1].content

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
            "messages": [response],
            "results": {question: retrieved_segments},
        }

    # Create graph
    workflow = StateGraph(GDPRState)

    # Add nodes
    workflow.add_node("agent", agent)
    retrieve = ToolNode([retriever_tool])
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

    agent = create_gdpr_agent(privacy_segments)
    results = {}

    for question in GDPR_QUESTIONS:
        inputs = {"messages": [("user", question)], "results": {}}

        output = agent.invoke(inputs)
        results[question] = output["results"][question]

    return results
