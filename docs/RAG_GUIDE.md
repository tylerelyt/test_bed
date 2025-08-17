# Guide: Retrieval-Augmented Generation (RAG) System

## 1. Overview

The Retrieval-Augmented Generation (RAG) system combines the strengths of a traditional TF-IDF based search engine with the generative power of Large Language Models (LLMs) via Ollama. It enhances user queries by first retrieving relevant documents and then feeding that context to an LLM to generate a comprehensive, context-aware answer.

## 2. Core Features & Modes

The system operates in three distinct modes, controlled via checkboxes in the UI.

### Mode 1: Direct LLM Chat (Retrieval Disabled)
- **How it works**: When the "Enable Retrieval (RAG)" checkbox is unchecked, the system sends the user's question directly to the LLM without any context.
- **Use Case**: Useful for general knowledge questions or creative tasks where the answer does not depend on the documents in the local index.

### Mode 2: Standard RAG (Retrieval Enabled)
- **How it works**: The system first retrieves the top-K most relevant documents from the index using TF-IDF. These documents are compiled into a context, which is prepended to the user's query in a prompt. The LLM then generates an answer based on this context.
- **Use Case**: This is the primary mode for answering questions based specifically on the content of the indexed documents.

### Mode 3: Multi-Step Reasoning (ReAct Style)
- **How it works**: When the "Enable Multi-step Reasoning" checkbox is checked, the system employs a ReAct (Reason + Act) style agent. The LLM thinks step-by-step, deciding whether to perform a `SEARCH` action to query the index or a `FINISH` action to provide the final answer.
- **Transparency**: In this mode, the "Prompt/Trace" output box displays the entire thought process of the agent, including its actions and the observations from the search tool.
- **Use Case**: Ideal for complex questions that may require synthesizing information from multiple documents or breaking down a problem into smaller parts.

## 3. How to Use

### Prerequisites
1.  **Ollama Service**: Ensure your local Ollama instance is running. The default URL is `http://localhost:11434`, which can be configured in `src/search_engine/config.py`.
2.  **LLM Models**: Make sure you have pulled the necessary models (e.g., `ollama pull llama3.1`). The default model is configured in `config.py`.

### Step-by-Step Guide
1.  **Navigate**: Go to the "🤖 RAG / Context Engineering" tab in the UI.
2.  **Check Connection**: Click the **"Check Ollama Connection"** button to verify that the system can communicate with Ollama and to see a list of available models. This will also refresh the model dropdown.
3.  **Enter Query**: Type your question into the "Enter your question" text box.
4.  **Select Mode**:
    - For **Standard RAG**, keep `Enable Retrieval (RAG)` checked.
    - For **Direct Chat**, uncheck `Enable Retrieval (RAG)`.
    - For **Multi-step Reasoning**, check both `Enable Retrieval (RAG)` and `Enable Multi-step Reasoning`.
5.  **Adjust Parameters (Optional)**:
    - **Retrieve Top-K**: Use the slider to control how many documents are retrieved for context.
    - **Select Model**: Choose a specific LLM from the dropdown list.
6.  **Execute**: Click the **"🚀 Execute Query"** button.

## 4. Understanding the Output

- **Generated Answer**: The final answer from the LLM.
- **Processing Info**: Shows the time taken, the model used, and the number of retrieved documents.
- **Prompt / Reasoning Trace**:
    - In **Direct Chat** or **Standard RAG** mode, this box shows the exact prompt sent to the LLM.
    - In **Multi-Step Reasoning** mode, this box displays the full chain-of-thought trace of the agent.
- **Retrieved Documents**: A table listing the documents retrieved from the index, along with their TF-IDF relevance scores.

## 5. Technical Implementation

### Standard RAG Flow
1.  The user's query is sent to `IndexService` to retrieve relevant documents.
2.  The content of these documents is concatenated to form a single `context` string.
3.  A prompt is constructed using a template that includes the `context` and the original `query`.
4.  This prompt is sent to the Ollama API endpoint for generation.

### Prompt Template (Standard RAG)
```
Based on the following context, please answer the user's question. If the context does not contain the relevant information, state that you cannot answer based on the provided information.

Context:
{context}

User Question: {query}

Please answer in English:
```

### ReAct Reasoning Flow
1.  The agent is given an initial prompt that includes the user's query and a description of the available tools (`SEARCH` and `FINISH`).
2.  The LLM generates a `Thought` and an `Action`.
3.  If the action is `SEARCH("some query")`, the system calls the `IndexService` and returns the results as an `Observation`.
4.  The `Thought`, `Action`, and `Observation` are appended to a scratchpad, and the loop continues until the LLM generates a `FINISH("final answer")` action.

This iterative process allows the model to reason and gather information dynamically before formulating a final answer. 