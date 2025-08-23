# Guide: Context Engineering System ([Back to README](../README.md))

## 1. Overview

The Context Engineering system augments user queries by retrieving relevant context from the local index and assembling a prompt for an LLM via Ollama. It supports direct LLM chat, context-enabled answering, and optional multi-step reasoning (ReAct-style).

## 2. Core Features & Modes

The UI exposes three modes via checkboxes and controls:

### Mode 1: Direct LLM Chat (Context Disabled)
- How it works: Sends the user's question directly to the LLM without retrieval.
- Use case: General knowledge questions or creative tasks not tied to local documents.

### Mode 2: Context Engineering (Retrieval Enabled)
- How it works: Retrieves top‑K relevant documents via TF‑IDF, assembles a context, and prepends it to the user's question in a prompt. The LLM answers based on this provided context.
- Use case: Answers grounded in the local indexed documents.

### Mode 3: Multi-Step Reasoning (ReAct Style)
- How it works: When enabled, the agent performs iterative reasoning with two actions: `SEARCH` (query index) and `FINISH` (output final answer). The UI shows the agent's steps in the trace box.
- Use case: Complex queries requiring decomposition or multi-document synthesis.

## 3. How to Use

### Prerequisites
1. Ollama service running at `http://localhost:11434` (configurable in code).
2. Required models pulled in Ollama (e.g., `ollama pull llama3.1`).

### Steps
1. Navigate to the "🤖 Context Engineering" tab.
2. Click "Check Ollama Connection" to verify connectivity and refresh the model list.
3. Enter your question.
4. Select options:
   - Keep "Enable Context Engineering" checked for retrieval‑augmented answering.
   - Enable "Multi-step Reasoning" for ReAct mode if needed.
   - Adjust Top‑K and select a model.
5. Click "🚀 Execute Query".

## 4. Understanding the Output

- Generated Answer: Final answer from the LLM.
- Processing Info: Time taken, model used, count of retrieved documents.
- Prompt / Reasoning Trace:
  - Direct or Context modes show the exact prompt sent to the LLM.
  - Multi-step mode shows the full chain‑of‑thought trace (Thought/Action/Observation).
- Retrieved Documents: Table of doc IDs and TF‑IDF scores; a separate box shows the assembled context.

## 5. Technical Implementation

### Context Flow (Retrieval Enabled)
1. `IndexService` retrieves relevant documents.
2. The documents are concatenated into a context string.
3. A prompt template embeds the context and the original query.
4. The prompt is sent to the Ollama generation API.

### Prompt Template (Context Mode)
```
Based on the following context, please answer the user's question. If the context does not contain the relevant information, state that you cannot answer based on the provided information.

Context:
{context}

User Question: {query}

Please answer in English:
```

### ReAct Reasoning Flow
1. The agent is given a description of tools (`SEARCH`, `FINISH`).
2. The LLM produces a Thought and an Action.
3. On `SEARCH("...")`, the system queries the index and returns an Observation.
4. The loop continues until `FINISH("final answer")`.

This enables step‑by‑step reasoning and evidence gathering before producing an answer.

