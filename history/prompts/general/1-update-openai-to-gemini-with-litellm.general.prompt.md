---
id: 1
title: Update OpenAI to Gemini with LiteLLM
stage: general
date: 2025-12-13
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: Aksystems
command: /sp.phr
labels: ["gemini", "litellm", "api-integration", "refactor"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\api\rag.py
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\tools\rag_tool.py
 - D:\github\user1\AI-SpecDriven-Development\Phase-2 Chatbot using Nextjs\backend\requirements.txt
tests:
 - none
---

## Prompt

Use the liteLLM library of OpenAI Agent SDK to use gemini as LLM in OPENAI Agnet SDK

## Response snapshot

Updated the codebase to use LiteLLM for routing OpenAI-compatible API calls to Google's Gemini API, allowing the retention of OpenAI for agent functionality while using Gemini for LLM completions.

## Outcome

- ✅ Impact: Successfully integrated LiteLLM to route LLM calls to Gemini API while keeping OpenAI for agent functionality
- 🧪 Tests: none
- 📁 Files: Updated rag.py, rag_tool.py, and requirements.txt
- 🔁 Next prompts: Test the updated API with Gemini API key
- 🧠 Reflection: LiteLLM provides an elegant solution to use different LLM providers with OpenAI-compatible SDK calls

## Evaluation notes (flywheel)

- Failure modes observed: Initially tried to replace OpenAI completely, but user clarified they wanted to keep OpenAI for agents while using Gemini for LLM
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Add proper error handling for API key configuration