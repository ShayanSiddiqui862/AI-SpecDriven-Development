# LLM Prompt Engineering Examples

This directory contains LLM prompt engineering examples for task decomposition in humanoid robotics applications.

## Overview

This package provides:
- Task decomposition prompts for complex robotics tasks
- Natural language to action mapping
- Context-aware prompting for humanoid behaviors
- Multi-modal instruction following

## Components

- `task_decomposer.py` - Task decomposition using LLMs
- `action_mapper.py` - Natural language to robot action mapping
- `context_aware.py` - Context-aware prompting system
- `prompt_templates.py` - Reusable prompt templates

## Usage

```bash
# Run task decomposition example
python3 task_decomposer.py --task "pick up red ball"

# Run action mapping example
python3 action_mapper.py --command "move to kitchen"
```

## Requirements

- OpenAI API key or local LLM
- Python 3.8+
- Appropriate LLM libraries