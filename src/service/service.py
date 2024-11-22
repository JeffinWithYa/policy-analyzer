import json
import logging
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, List, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient
from pydantic import BaseModel

from agents import DEFAULT_AGENT, agents
from agents.gdpr_compliance_agent import analyze_gdpr_compliance
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.")),
    ],
) -> None:
    if http_auth.credentials != os.getenv("AUTH_SECRET"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


bearer_depend = [Depends(verify_bearer)] if os.getenv("AUTH_SECRET") else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    # TODO: It's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        for a in agents.values():
            a.checkpointer = saver
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=bearer_depend)


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    }
    return kwargs, run_id


async def ainvoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the default agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    logger.info(f"Received invoke request - Message: {user_input.message[:200]}...")
    logger.info(f"Using model: {user_input.model}")
    logger.info(f"Thread ID: {user_input.thread_id}")

    response = await ainvoke(user_input=user_input)

    logger.info(f"Returning response - Content: {response.content[:200]}...")
    return response


@router.post("/{agent_id}/invoke")
async def agent_invoke(user_input: UserInput, agent_id: str) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    return await ainvoke(user_input=user_input, agent_id=agent_id)


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        new_messages = []
        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]

        # Also yield intermediate messages from agents.utils.CustomData.adispatch().
        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get(
            "tags", []
        ):
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if (
                chat_message.type == "human"
                and chat_message.content == user_input.message
            ):
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream(user_input: StreamInput) -> StreamingResponse:
    """
    Stream the default agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input), media_type="text/event-stream"
    )


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
async def agent_stream(user_input: StreamInput, agent_id: str) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id=agent_id), media_type="text/event-stream"
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = agents[DEFAULT_AGENT]
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [
            langchain_to_chat_message(m) for m in messages
        ]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


# Add response model
class GDPRAnalysisResult(BaseModel):
    results: Dict[str, List[Dict[str, Any]]]


class GDPRRequest(BaseModel):
    privacy_segments: List[Dict[str, Any]]


@router.post("/gdpr/analyze", response_model=GDPRAnalysisResult)
async def analyze_gdpr(request: GDPRRequest) -> GDPRAnalysisResult:
    """
    Analyze privacy policy segments for GDPR compliance.
    """
    try:
        logger.info("Received GDPR analysis request")
        logger.debug(f"Request data: {request.model_dump()}")

        # Validate privacy segments format
        for segment in request.privacy_segments:
            if (
                not isinstance(segment, dict)
                or "segment" not in segment
                or "model_analysis" not in segment
            ):
                logger.error(f"Invalid segment format: {segment}")
                raise HTTPException(
                    status_code=400,
                    detail="Each segment must contain 'segment' and 'model_analysis' fields",
                )

        logger.info(f"Processing {len(request.privacy_segments)} privacy segments")

        # Get or create the GDPR agent
        from agents.agents import get_gdpr_agent

        agent = get_gdpr_agent()

        results = analyze_gdpr_compliance(request.privacy_segments)

        logger.info("Analysis complete")
        logger.debug(f"Results: {results}")

        return GDPRAnalysisResult(results=results)

    except Exception as e:
        logger.error(f"An error occurred during GDPR analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Unexpected error during GDPR analysis"
        )


app.include_router(router)
