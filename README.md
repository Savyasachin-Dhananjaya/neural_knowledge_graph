# Neural Knowledge Graph Server 🧠

**A Semantic Memory System for Local AI Agents**

This project is a standalone, CPU-optimized **Knowledge Graph Operating System** designed to act as the long-term memory ("Hippocampus") for local Large Language Models (LLMs) like Llama 3 or Gemma.

Unlike traditional vector databases that chunk text blindly, this system uses a **Graph + Neural Reranking** approach. It stores structured Entities and atomic Observations, then uses an embedded neural engine ([WordLlama](https://github.com/dleemiller/WordLlama)) to semantically rank and retrieve memories based on relevance to a specific thought.

## 🌟 Key Features

* **🧠 Embedded Neural Reranking:** Uses `wordllama` (Llama 3 70B distilled or Custom Fine-tunes) to perform state-of-the-art semantic reranking entirely on the CPU. No external APIs required.
* **🕸️ Graph Architecture:** Stores data as **Entities** (Nodes), **Relations** (Edges), and **Observations** (Atomic Facts).
* **🏷️ Tag-Based Aggregation:** Retrieves context by "Broad Domain" (e.g., Professional, Personal) rather than fragile keyword searches.
* **⏳ Temporal Provenance:** Automatically tracks `created_at` (Immutable) vs `last_modified` (Mutable) timestamps. This allows the LLM to reconstruct timelines and understand the history of a fact.
* **🔒 100% Local & Private:** Runs as a lightweight FastAPI service on your local machine (Ubuntu/Linux). Perfect for confidential legal or personal data.
* **🛠️ Full CRUD:** Support for creating, deleting, and searching entities, relations, and specific observations.

## 📐 Data Strategy (Critical)

**⚠️ Important:** While the API allows any string for `entityType`, it is best to work with a **Fixed Taxonomy**.

To ensure the Neural Reranker can aggregate all relevant facts, you should restrict your system to a limit set of **Canonical Entity Types**. Do not create new types on the fly.

**Recommended Taxonomy:**

1. **`Personal`**: Facts about the user, family, health, and biometrics.
2. **`Professional`**: Work, clients, cases, projects, and business entities.
3. **`System_Server`**: Technical configurations, hardware specs, software preferences.
4. **`Events`**: Significant milestones or historical dates.

*Why?* When the AI asks "What server software do I use?", it queries the `System_Server` tag. If you split your data into `Linux`, `Coding`, and `Config` tags, the AI will miss half the context.

## 🏗️ Architecture

1. **Storage Layer:** A single, portable `memory.json` file.
2. **Logic Layer:** FastAPI server handling CRUD operations and timestamp logic.
3. **Neural Layer:** WordLlama loaded in RAM. It converts text to 1024-dimensional vectors and calculates cosine similarity to rank memories against the user's query.

## 🚀 Installation

### 1. Prerequisites

* Python 3.10+
* Ubuntu / Linux (Recommended)

### 2. Setup Environment

```
# Clone the repo
git clone https://github.com/Savyasachin-Dhananjaya/neural_knowledge_graph
cd neural_knowledge_graph

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Model Configuration

The server supports two modes. Edit `main.py` to select one:

* **Standard Mode (Recommended):** Uses `l3_supercat` (Llama 3 distilled). Auto-downloads on first run.
```
WL_CONFIG = "l3_supercat"

```


* **Custom Mode (Advanced):** Uses your fine-tuned `wordllama` weights. Requires `tokenizer.json` and `.safetensors` file in the root directory.

## 🏃‍♂️ Usage

### Manual Start

```
# Run on port 9041
uvicorn main:app --host 0.0.0.0 --port 9041

```

### Run as System Service (Background)

1. Create the service file:
```
sudo nano /etc/systemd/system/memory_graph.service

```

2. Paste the configuration (ensure paths match your user):
```
[Unit]
Description=Neural Knowledge Graph (Port 9041)
After=network.target

[Service]
User=your-user
WorkingDirectory=/home/your-user/neural_knowledge_graph
ExecStart=/home/your-user/neural_knowledge_graph/venv/bin/uvicorn main:app --host 0.0.0.0 --port 9041
Restart=on-failure
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target

```


3. Enable and start:
```
sudo systemctl enable --now memory_graph.service

```



## 🔌 API Playbook

| Action | Method | Command / Payload |
| --- | --- | --- |
| **Check Status** | | `curl http://localhost:9041/docs` |
| **Create Entity** | `POST /create_entities` | `{"entities": [{"name": "Case 404", "entityType": "Professional"}]}` |
| **Add Fact** | `POST /add_observations` | `{"observations": [{"entityName": "Case 404", "contents": ["Hearing on Jan 8th."]}]}` |
| **Link Entities** | `POST /create_relations` | `{"relations": [{"from": "User", "to": "Case 404", "relationType": "counsel"}]}` |
| **Neural Search** | `POST /retrieve_context` | `{"canonical_entity": "Professional", "query_thought": "When is the hearing?", "top_k": 1}` |
| **Dump Graph** | `GET /read_graph` | |

## 🤖 LLM Integration (System Prompt)

To make your AI assistant effective, give it these instructions. Note the strict rule on Entity Types.

> **ROLE:** You are the Knowledge Manager for the Neural Memory System.
> 
> **PROTOCOL:**
> 1. **Atomic & Self-Contained:** "User's office opens at 9 AM" (Good) vs "It opens at 9 AM" (Bad).
> 2. **Canonical Taxonomy:** You must categorize all entities into one of these 4 types: `["Personal", "Professional", "System_Server", "Events"]`. Do not invent new types.
> 
> 
> **TOOLS:**
> 1. `retrieve_context(canonical_entity, query_thought)`: Use this first to check facts. `canonical_entity` must be one of the 4 Allowed Types.
> 2. `add_observations(observations)`: Use this to store *new* or *changed* facts.
> 3. `create_entities(entities)`: Use this only for major new topics.
>
> 

## 📂 Data Structure (`memory.json`)

Your data is stored in a clean, human-readable JSON format.

```json
{
  "entities": [
    {
      "name": "Some Name",
      "entityType": "Professional",
      "created_at": "YYYY-MM-DDThh:mm:ss",
      "last_modified": YYYY-MM-DDThh:mm:ss",
      "observations": [
        {
          "content": "User prefers X over Y.",
          "timestamp": YYYY-MM-DDThh:mm:ss"
        }
      ]
    }
  ],
  "relations": [
    {
      "from": "Some Name",
      "to": "Some Related Entity",
      "relationType": "Some relation between Them",
      "created_at": YYYY-MM-DDThh:mm:ss"
    }
  ]
}

```
