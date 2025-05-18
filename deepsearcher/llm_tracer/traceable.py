import functools
from typing import Any, Callable, Literal, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def lazy_traceable(
    run_type: Literal[
        "tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"
    ] = "chain",
    *,
    name: str = None,
    tags: list[str] = None,
    metadata: dict = None,
) -> Callable[[F], F]:
    """
    A decorator that decides whether to use the original function or the decorated function based on LangSmith availability.

    Args:
        run_type: The LangSmith run type, options include:
            - 'tool': For utility functions that perform specific operations
            - 'chain': For composite operations that combine multiple steps (default)
            - 'llm': For direct language model API calls
            - 'retriever': For document/information retrieval operations
            - 'embedding': For vector embedding generation operations
            - 'prompt': For prompt template rendering
            - 'parser': For output parsing operations
        name: Optional custom name for the traced run
        tags: Optional list of tags to categorize the run
        metadata: Optional dictionary of metadata to attach to the run

    Returns:
        A decorator that uses the LangSmith traceable decorator if installed, otherwise returns the original function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        try:
            from langsmith.run_helpers import traceable as langsmith_traceable

            return langsmith_traceable(run_type=run_type, name=name, tags=tags, metadata=metadata)(
                func
            )
        except ImportError:
            return wrapper

    return decorator
