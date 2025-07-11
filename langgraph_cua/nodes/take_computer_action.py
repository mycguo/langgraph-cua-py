import time
from typing import Any, Dict, Optional

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from scrapybara.types import ComputerResponse, InstanceGetStreamUrlResponse

from ..types import CUAState, get_configuration_with_defaults
from ..utils import get_instance, is_computer_tool_call

# Copied from the OpenAI example repository
# https://github.com/openai/openai-cua-sample-app/blob/eb2d58ba77ffd3206d3346d6357093647d29d99c/computers/scrapybara.py#L10
CUA_KEY_TO_SCRAPYBARA_KEY = {
    "/": "slash",
    "\\": "backslash",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "capslock": "Caps_Lock",
    "cmd": "Meta_L",
    "delete": "Delete",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "home": "Home",
    "insert": "Insert",
    "option": "Alt_L",
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "tab": "Tab",
    "win": "Meta_L",
}


def take_computer_action(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Executes computer actions based on the tool call in the last message.

    Args:
        state: The current state of the CUA agent.
        config: The runnable configuration.

    Returns:
        A dictionary with updated state information.
    """
    message: AnyMessage = state.get("messages", [])[-1]
    assert message.type == "ai", "Last message must be an AI message"
    
    # For Claude/Anthropic, tool calls are in the tool_calls attribute
    tool_calls = getattr(message, 'tool_calls', [])
    
    if not tool_calls:
        raise ValueError("Cannot take computer action without a computer call in the last message.")
    
    # Find the computer tool call
    computer_tool_call = None
    for tool_call in tool_calls:
        if tool_call.get('name') == 'computer':
            computer_tool_call = tool_call
            break
    
    if not computer_tool_call:
        raise ValueError("No computer tool call found in the last message.")

    instance_id = state.get("instance_id")
    if not instance_id:
        raise ValueError("Instance ID not found in state.")
    instance = get_instance(instance_id, config)

    configuration = get_configuration_with_defaults(config)
    environment = configuration.get("environment")
    auth_state_id = configuration.get("auth_state_id")
    authenticated_id = state.get("authenticated_id")

    if (
        environment == "web"
        and auth_state_id is not None
        and (
            (authenticated_id is None)
            or (authenticated_id is not None and authenticated_id != auth_state_id)
        )
    ):
        instance.authenticate(auth_state_id=auth_state_id)
        authenticated_id = auth_state_id

    stream_url: Optional[str] = state.get("stream_url")
    if not stream_url:
        # If the stream_url is not yet defined in state, fetch it, then write to the custom stream
        # so that it's made accessible to the client (or whatever is reading the stream) before any actions are taken.
        stream_url_response: InstanceGetStreamUrlResponse = instance.get_stream_url()
        stream_url = stream_url_response.stream_url

        writer = get_stream_writer()
        writer({"stream_url": stream_url})

    # Extract action from Claude tool call format
    action = computer_tool_call.get('args', {})
    tool_call_id = computer_tool_call.get('id')
    tool_message: Optional[ToolMessage] = None

    try:
        computer_response: Optional[ComputerResponse] = None
        action_type = action.get("action")

        if action_type == "left_click":
            computer_response = instance.computer(
                action="click_mouse",
                button="left",
                coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]],
            )
        elif action_type == "right_click":
            computer_response = instance.computer(
                action="click_mouse",
                button="right",
                coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]],
            )
        elif action_type == "middle_click":
            computer_response = instance.computer(
                action="click_mouse",
                button="middle",
                coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]],
            )
        elif action_type == "double_click":
            computer_response = instance.computer(
                action="click_mouse",
                button="left",
                coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]],
                num_clicks=2,
            )
        elif action_type == "left_click_drag":
            computer_response = instance.computer(
                action="drag_mouse",
                path=[[action.get("startCoordinate")[0], action.get("startCoordinate")[1]], 
                      [action.get("endCoordinate")[0], action.get("endCoordinate")[1]]],
            )
        elif action_type == "key":
            # Handle both "key" and "text" fields for key actions
            key_value = action.get("key") or action.get("text")
            if key_value:
                mapped_keys = [
                    CUA_KEY_TO_SCRAPYBARA_KEY.get(key_value.lower(), key_value.lower())
                ]
                computer_response = instance.computer(action="press_key", keys=mapped_keys)
            else:
                raise ValueError("Key action missing key or text parameter")
        elif action_type == "mouse_move":
            computer_response = instance.computer(
                action="move_mouse", coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]]
            )
        elif action_type == "screenshot":
            computer_response = instance.computer(action="take_screenshot")
        elif action_type == "wait":
            # Sleep for the specified duration (in seconds)
            duration = action.get("duration", 2)
            time.sleep(duration)
            # Take a screenshot after waiting
            computer_response = instance.computer(action="take_screenshot")
        elif action_type == "scroll":
            computer_response = instance.computer(
                action="scroll",
                delta_x=action.get("coordinate")[0] if action.get("direction") == "left" else (
                    -action.get("coordinate")[0] if action.get("direction") == "right" else 0),
                delta_y=action.get("coordinate")[1] if action.get("direction") == "down" else (
                    -action.get("coordinate")[1] if action.get("direction") == "up" else 0),
                coordinates=[action.get("coordinate")[0], action.get("coordinate")[1]],
            )
        elif action_type == "type":
            computer_response = instance.computer(action="type_text", text=action.get("text"))
        else:
            raise ValueError(f"Unknown computer action received: {action}")

        if computer_response:
            output_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": computer_response.base_64_image,
                },
            }
            tool_message = ToolMessage(
                content=[output_content],
                tool_call_id=tool_call_id,
                additional_kwargs={"type": "computer_call_output"},
            )
    except Exception as e:
        print(f"\n\nFailed to execute computer call: {e}\n\n")
        print(f"Computer call details: {computer_tool_call}\n\n")

    return {
        "messages": tool_message if tool_message else None,
        "instance_id": instance.id,
        "stream_url": stream_url,
        "authenticated_id": authenticated_id,
    }
