from typing import Any, Union

from langchain_core.runnables import RunnableConfig
from scrapybara import Scrapybara
from scrapybara.client import BrowserInstance, UbuntuInstance, WindowsInstance

from .types import get_configuration_with_defaults


def get_scrapybara_client(api_key: str) -> Scrapybara:
    """
    Gets the Scrapybara client, using the API key provided.

    Args:
        api_key: The API key for Scrapybara.

    Returns:
        The Scrapybara client.
    """
    if not api_key:
        raise ValueError(
            "Scrapybara API key not provided. Please provide one in the configurable fields, "
            "or set it as an environment variable (SCRAPYBARA_API_KEY)"
        )
    client = Scrapybara(api_key=api_key)
    return client


def get_instance(
    id: str, config: RunnableConfig
) -> Union[UbuntuInstance, BrowserInstance, WindowsInstance]:
    """
    Gets an instance by its ID from Scrapybara.

    Args:
        id: The ID of the instance to get.
        config: The configuration for the runnable.

    Returns:
        The instance.
    """
    configuration = get_configuration_with_defaults(config)
    scrapybara_api_key = configuration.get("scrapybara_api_key")
    client = get_scrapybara_client(scrapybara_api_key)
    return client.get(id)


def is_computer_tool_call(tool_calls: Any) -> bool:
    """
    Checks if the given tool calls contain a computer call.

    Args:
        tool_calls: The tool calls to check.

    Returns:
        True if the tool calls contain a computer call, false otherwise.
    """
    if not tool_calls or not isinstance(tool_calls, list):
        return False

    return any(call.get("name") == "computer" for call in tool_calls)
