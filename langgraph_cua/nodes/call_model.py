from typing import Any, Dict, Optional, Union

from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from anthropic import Anthropic

from ..types import CUAState, get_configuration_with_defaults


# Scrapybara does not allow for configuring this. Must use a hardcoded value.
DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 768


def _prompt_to_sys_message(prompt: Union[str, SystemMessage, None]):
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return {"role": "system", "content": prompt}
    return prompt


async def call_model(state: CUAState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invokes the computer preview model with the given messages.

    Args:
        state: The current state of the thread.

    Returns:
        The updated state with the model's response.
    """
    configuration = get_configuration_with_defaults(config)
    prompt = _prompt_to_sys_message(configuration.get("prompt"))
    messages = state.get("messages", [])
    
    # Convert LangChain messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'human':
            anthropic_messages.append({"role": "user", "content": msg.content})
        elif hasattr(msg, 'type') and msg.type == 'ai':
            # Handle AI messages with tool calls
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            
            # Add tool_use blocks if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["args"]
                    })
            
            anthropic_messages.append({
                "role": "assistant",
                "content": content
            })
        elif hasattr(msg, 'type') and msg.type == 'tool':
            # Handle tool messages
            tool_content = []
            if hasattr(msg, 'content') and msg.content:
                if isinstance(msg.content, list):
                    tool_content = msg.content
                else:
                    tool_content = [{"type": "text", "text": str(msg.content)}]
            
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": tool_content}]
            })

    client = Anthropic()
    
    tools = [{
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": DEFAULT_DISPLAY_WIDTH,
        "display_height_px": DEFAULT_DISPLAY_HEIGHT,
    }]
    
    if prompt and prompt.get("content"):
        system_prompt = [{"type": "text", "text": prompt.get("content")}]
    else:
        system_prompt = None
    
    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
            messages=anthropic_messages,
            tools=tools,
            betas=["computer-use-2025-01-24"]
        )
        
        # Convert response back to LangChain format
        content = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input
                })
        
        # Create AIMessage with tool calls
        ai_message = AIMessageChunk(
            content=" ".join(content) if content else "",
            tool_calls=tool_calls
        )
        
        return {
            "messages": ai_message,
        }
        
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        raise
