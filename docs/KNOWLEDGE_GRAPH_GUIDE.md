# 🕸️ Guide: Knowledge Graph (KG) System

## 1. Overview

The Knowledge Graph (KG) system uses a Large Language Model (LLM) to perform Named Entity Recognition (NER) on your documents. It extracts entities (like people, places, and concepts) and the relationships between them, building a graph-based index of your content. This allows you to explore the connections within your document set.

## 2. Core Features

### 🧠 LLM-Based Named Entity Recognition (NER)
- **Engine**: Uses an Ollama-hosted LLM (or a compatible OpenAI API) to identify entities and relationships. The default model is configurable in `src/search_engine/config.py`.
- **Entity Types**: Can identify types such as Person, Location, Organization, Concept, Technology, Product, and Event.
- **Relation Types**: Can identify relationships like `belongs_to`, `located_in`, `develops`, `uses`, `related_to`, and `influences`.

### 🕸️ Graph Construction
- **Backend**: Uses the `networkx` library to build a directed multi-graph, allowing multiple distinct relationships between the same two entities.
- **Functionality**: The system stores entities and their relationships, automatically deduplicates them, and links them back to the source documents.

### 🔍 Entity Exploration
- **Entity Search**: Find specific entities within the graph by name.
- **Relation Viewing**: Once an entity is found, you can view all its incoming and outgoing relationships, showing how it connects to other entities.
- **Source Document Linking**: See which documents an entity was extracted from.

## 3. How to Use

The entire workflow is managed from the **"Index & KG"** tab.

### Step 1: Build the Knowledge Graph
1.  Navigate to the **"Index & KG"** tab in the main UI.
2.  Go to the **"🕸️ Knowledge Graph Management"** sub-tab.
3.  **Select a Model**: Choose an appropriate NER model from the dropdown. Larger models are more accurate but slower.
4.  **Build Graph**: Click the **"🔨 Build Knowledge Graph"** button. This process can be time-consuming, as it involves sending document content to the LLM for analysis. Monitor the progress in the console.
5.  **Confirmation**: A status message will appear when the build is complete.

### Step 2: View Graph Statistics
- After building, click the **"📊 Refresh Stats"** button to see:
    - Total number of entities and relations.
    - A breakdown of entities and relations by type.

### Step 3: Search for Entities
1.  Enter a keyword in the **"Search for an entity..."** text box.
2.  Click **"🔍 Search Entities"**.
3.  A table will appear showing matching entities, their type, description, and how many documents they appear in.

### Step 4: Explore Entity Details
1.  After searching, enter the exact name of an entity from the search results into the **"Enter exact entity name to see details..."** text box.
2.  Click **"📄 Get Entity Details"**.
3.  The UI will display detailed information about that entity, including:
    - **Outgoing Relations**: What this entity *does to* other entities.
    - **Incoming Relations**: What other entities *do to* this one.
    - **Related Documents**: A list of document IDs where this entity was found.

## 4. Technical Implementation

### NER Process
1.  **Chunking**: Long documents are automatically split into smaller chunks to fit the LLM's context window.
2.  **LLM Extraction**: A detailed prompt instructs the LLM to identify entities and relations and return them in a structured JSON format.
3.  **JSON Parsing**: The system parses the LLM's JSON output.
4.  **Deduplication**: Results from all chunks are aggregated, and duplicate entities and relations are merged.

### Graph Storage
- The graph is built as a `networkx.MultiDiGraph` object.
- **Nodes**: Represent entities, with attributes like `entity_type` and `description`.
- **Edges**: Represent relationships, with attributes like `predicate`.
- The entire graph object is serialized and saved to `models/knowledge_graph.pkl` using Python's `pickle` module.

## 5. Important Notes & Troubleshooting

### Model Selection
- **`llama3.1:8b`**: A good balance of performance and accuracy.
- **`qwen2.5:7b`**: Optimized for Chinese language tasks.
- Smaller models will be faster but may produce less accurate or less structured NER results.

### Performance
- Building the KG is a CPU and/or GPU-intensive task that can take a long time, depending on the number of documents and the LLM's speed.
- It is recommended to run this on a powerful machine and to test with a small subset of documents first.

### Common Issues
1.  **Build Fails**: Check that the Ollama service is running and the selected model is available. Look for detailed error messages in the console.
2.  **Inaccurate Extraction**: The quality of the KG depends heavily on the LLM's ability to follow instructions and perform NER. If results are poor, try a different, more powerful model.
3.  **No Search Results**: Ensure the graph has been successfully built and contains the entities you are searching for. 