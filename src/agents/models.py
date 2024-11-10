import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models: dict[str, BaseChatModel] = {}
if os.getenv("OPENAI_API_KEY") is not None:
    models["gpt-4o-mini"] = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.5, streaming=True
    )
    models["gpt-4o"] = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
    models["gpt-4"] = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, streaming=True)
    models["gpt-3.5"] = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0.5, streaming=True
    )
    models["o1-preview"] = ChatOpenAI(
        model="o1-preview", temperature=0.5, streaming=True
    )
    models["o1-mini"] = ChatOpenAI(model="o1-mini", temperature=0.5, streaming=True)
if os.getenv("GROQ_API_KEY") is not None:
    models["llama-3.1-70b-versatile"] = ChatGroq(
        model="llama-3.1-70b-versatile", temperature=0.5
    )
    models["llama-3.1-8b-instant"] = ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.5
    )
    models["mixtral-8x7b-32768"] = ChatGroq(
        model="llama-3.1-405b-reasoning", temperature=0.5
    )
if os.getenv("GOOGLE_API_KEY") is not None:
    models["gemini-1.5-flash"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0.5, streaming=True
    )
    models["gemini-1.5-flash-8b"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b", temperature=0.5, streaming=True
    )
    models["gemini-1.5-pro"] = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0.5, streaming=True
    )
if os.getenv("ANTHROPIC_API_KEY") is not None:
    models["claude-3-haiku"] = ChatAnthropic(
        model="claude-3-haiku-20240307", temperature=0.5, streaming=True
    )

if not models:
    print("No LLM available. Please set API keys to enable at least one LLM.")
    if os.getenv("MODE") == "dev":
        print("FastAPI initialized failed. Please use Ctrl + C to exit uvicorn.")
    exit(1)
