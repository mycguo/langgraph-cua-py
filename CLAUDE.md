# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is LangGraph Computer Use Agent (CUA) - a Python library for creating computer use agents using LangGraph. The agent can interact with virtual machines through Scrapybara to perform computer automation tasks.

## Architecture

### Core Components

- **`langgraph_cua/graph.py`**: Main graph definition with workflow nodes and routing logic
- **`langgraph_cua/nodes/`**: Individual workflow nodes:
  - `call_model.py`: Invokes OpenAI's computer-use-preview model
  - `create_vm_instance.py`: Creates Scrapybara VM instances
  - `take_computer_action.py`: Executes computer actions on VM
- **`langgraph_cua/types.py`**: Type definitions and state schemas
- **`langgraph_cua/utils.py`**: Utility functions for Scrapybara client and tool detection

### State Management

The agent uses `CUAState` which tracks:
- `messages`: Conversation history with add_messages reducer
- `instance_id`: Scrapybara VM instance ID
- `stream_url`: Live stream URL for VM viewing
- `authenticated_id`: Auth state for persistent sessions

### Workflow Logic

1. **call_model**: Invokes OpenAI model with computer_use_preview tool
2. **create_vm_instance**: Creates VM if needed when computer tool is called
3. **take_computer_action**: Executes computer actions on the VM
4. **Routing**: `take_action_or_end` and `reinvoke_model_or_end` control flow

## Development Commands

### Setup
```bash
uv venv
source .venv/bin/activate
uv sync --all-groups
```

### Testing
```bash
make test                    # Run all tests
make test TEST_FILE=path     # Run specific test file
pytest -xvs tests/integration/test_cua.py  # Run integration tests
```

### Linting and Formatting
```bash
make lint                    # Check code style
make format                  # Auto-format code
```

### Environment Variables
```bash
export OPENAI_API_KEY=<key>
export SCRAPYBARA_API_KEY=<key>
```

## Configuration

The `create_cua()` function accepts:
- `scrapybara_api_key`: Scrapybara API key (or SCRAPYBARA_API_KEY env var)
- `timeout_hours`: VM timeout (0.01-24 hours, default: 1)
- `zdr_enabled`: Zero Data Retention mode (default: False)
- `auth_state_id`: Persistent authentication state ID
- `environment`: VM environment ("web", "ubuntu", "windows")
- `prompt`: System prompt for the agent
- `recursion_limit`: Max recursive calls (default: 100)

## Testing Strategy

- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Integration tests require valid API keys and network access
- Use `pytest --disable-socket --allow-unix-socket` for unit tests

## Dependencies

- **LangGraph**: Core workflow framework
- **Scrapybara**: VM management and computer use
- **OpenAI**: Language model with computer_use_preview
- **uv**: Package management and virtual environments