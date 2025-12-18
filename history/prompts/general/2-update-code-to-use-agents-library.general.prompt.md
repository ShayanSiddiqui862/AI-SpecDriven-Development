---
id: 2
title: Update code to use agents library
stage: general
date: 2025-12-13
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: Aksystems
command: manual
labels: ["agents", "openai-agents", "config", "refactor"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\connection.py
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\requirements.txt
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\api\rag.py
tests:
 - none
---

## Prompt

i have created a connection.py file in the backend folder. You update the code and only use config and openai-agents library for creating agents

## Response snapshot

Updated the codebase to properly use the agents library for creating and managing agents, with proper configuration through the connection.py file. Updated requirements.txt to include the agents library and modified api/rag.py to use agent instances instead of direct LLM calls.

## Outcome

- ✅ Impact: Successfully updated code to use agents library with proper configuration pattern
- 🧪 Tests: none
- 📁 Files: Updated connection.py, requirements.txt, and api/rag.py
- 🔁 Next prompts: Test the agent functionality with the new implementation
- 🧠 Reflection: Proper abstraction of agent creation through connection.py enables better maintainability

## Evaluation notes (flywheel)

- Failure modes observed: Had to account for different possible response formats from the agents library
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Add proper error handling for agent creation and execution