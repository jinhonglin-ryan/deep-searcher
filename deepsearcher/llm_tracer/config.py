import os
from typing import Any

LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_PROJECT = "default"


def configure_langsmith() -> bool:
    """
    Configure LangSmith tracing based on environment variables.

    Returns:
        bool: True if LangSmith was successfully configured, False otherwise
    """
    try:
        api_key = os.getenv("LANGSMITH_API_KEY")
        project = os.getenv("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

        if not api_key:
            print("No LangSmith API key provided. Tracing will not be enabled.")
            return False

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_PROJECT"] = project

        print(f"LangSmith tracing enabled for project: {project}")
        return True

    except ImportError:
        print(
            "LangSmith was requested but the package is not installed. "
            "Please install it with: pip install langsmith"
        )
        return False


def wrap_client(client: Any, client_type: str = "openai") -> Any:
    """
    Wraps an LLM client with LangSmith tracing capabilities.

    Args:
        client: The LLM client to wrap (OpenAI, Anthropic, etc.)
        client_type: The type of client to wrap ('openai', 'anthropic', etc.)

    Returns:
        The wrapped client if LangSmith is available, otherwise the original client
    """
    try:
        from langsmith.wrappers import wrap_anthropic, wrap_openai

        if client_type.lower() == "openai":
            return wrap_openai(client)
        elif client_type.lower() == "anthropic":
            return wrap_anthropic(client)
        else:
            print(f"Unsupported client type: {client_type}. Returning original client.")
            return client
    except ImportError:
        return client
