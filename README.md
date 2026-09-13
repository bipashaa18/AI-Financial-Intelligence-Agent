# 💰 AI Financial Intelligence Agent

An agentic financial RAG system that combines **advanced document retrieval, live market data, financial calculations, and conversational memory** in a single Streamlit application.

The project is designed as a portfolio/placement-ready demonstration of **RAG, agent orchestration, tool calling, vector databases, local LLMs, and production-style UI integration**.

---

## 🚀 What It Does

The agent can:

- Search **22 institutional financial PDFs** using a local RAG pipeline.
- Retrieve diverse results with **MMR (Maximal Marginal Relevance)**.
- Fetch current market information through **Yahoo Finance / yfinance**.
- Perform financial calculations such as:
  - CAGR
  - Sharpe ratio
  - DCF valuation
  - Compound interest
- Orchestrate tool calls using **LangGraph**.
- Persist conversation state using **SQLite checkpointing**.
- Provide an interactive **Streamlit chat interface**.
- Run locally using **Ollama**, avoiding paid LLM APIs.

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │     Streamlit UI     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    LangGraph Agent   │
                         │   Rule-based Router   │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
        ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐
        │ Financial RAG  │  │ Market Data    │  │ Calculator      │
        │                │  │                │  │                 │
        │ MMR Retrieval  │  │ yfinance       │  │ CAGR            │
        │ Source/page    │  │ Live prices    │  │ Sharpe          │
        │ citations      │  │ P/E, beta etc. │  │ DCF             │
        │ ChromaDB       │  │                │  │ Compound Interest│
        └───────┬────────┘  └────────────────┘  └─────────────────┘
                │
                ▼
       ┌─────────────────────┐
        │ 22 Financial PDFs    │
       │ RBI / IMF / OECD /   │
       │ Gold / Oil / Reports │
       └─────────────────────┘

                  ┌─────────────────────┐
                  │ SQLite Checkpointer │
                  │ Conversation Memory │
                  └─────────────────────┘
```

---

## 🔎 RAG Pipeline

The production path in `app.py` uses ChromaDB with MMR retrieval. MMR balances relevance with diversity so the agent receives useful, less-redundant document chunks. Each indexed chunk keeps its source filename and PDF page metadata for citations.

The notebooks contain earlier experiments with HyDE and RAG Fusion. Those experiments are useful for learning and comparison, but they are not required to run the Streamlit application.

---

## 🤖 Agentic Workflow

The system is orchestrated with LangGraph.

```text
START
  │
  ▼
route_question
  │
  ├── Financial / market / calculation question
  │              │
  │              ▼
  │            agent
  │              │
  │          tool calls?
  │           /       \
  │         yes        no
  │          │          │
  │          ▼          ▼
  │        tools   collect_tool_output
  │          │          │
  │          └──→ agent │
  │                     ▼
  │                  generate
  │                     │
  └─────────────────────┘
                        ▼
                       END
```

The agent currently has three tools:

| Tool                    | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `search_financial_docs` | Search the internal financial document collection    |
| `get_market_data`       | Retrieve current market information through yfinance |
| `financial_calculator`  | Perform financial calculations                       |

---

## 📚 Data

The RAG corpus contains **22 institutional financial PDFs**, including material related to:

- RBI
- IMF
- OECD
- Gold
- Oil
- Economic surveys
- Inflation
- GDP
- Monetary and fiscal policy

The documents are chunked and indexed in ChromaDB with source metadata so generated answers can cite the source file and page.

The verified local index contains **30,980 indexed chunks** after running `python ingest.py`.

---

## 🧰 Tech Stack

| Component           | Technology                       |
| ------------------- | -------------------------------- |
| LLM                 | Ollama `llama3.2`                |
| Embeddings          | Ollama `nomic-embed-text`        |
| RAG                 | LangChain                        |
| Agent orchestration | LangGraph                        |
| Vector database     | ChromaDB                         |
| Retrieval           | MMR + HyDE + RAG Fusion          |
| Market data         | yfinance                         |
| Memory              | SQLite + LangGraph checkpointing |
| UI                  | Streamlit                        |
| Language            | Python                           |

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AI-Financial-Intelligence-Agent.git
cd AI-Financial-Intelligence-Agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Windows (PowerShell):

```powershell
winget install --id Ollama.Ollama -e
```

Restart PowerShell after installation, then make sure Ollama is running:

```bash
ollama serve
```

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Build the vector database

Build the ChromaDB index from the PDFs included in this repository. Keep Ollama running while this command embeds the documents.

```powershell
python ingest.py
```

To rebuild an existing index from scratch, use `python ingest.py --reset`.

When cloning the GitHub repository, check whether `vectorstore/chroma/` is already present. If it is included in the repository, skip ingestion; the uploaded index is ready to use. Run `python ingest.py` only when the index is missing or you have changed the source PDFs.

### 6. Start Streamlit

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

### 7. Stop the services

Press `Ctrl+C` in the Streamlit and Ollama terminals when you are finished.

---

## 💬 Example Queries

Try questions such as:

```text
What is the current gold price?

What is the Nifty 50 level today?

What does the RBI report say about inflation?

What is the IMF GDP forecast?

Calculate CAGR: 50000 to 120000 in 8 years

What is AAPL stock price and P/E ratio?

Oil market outlook from the EIA report?

What are the key risk factors in the reports?
```

The application uses the following tool automatically based on your question:

- Internal report questions use `search_financial_docs`.
- Current prices and market metrics use `get_market_data`.
- Numerical questions use `financial_calculator`.

For calculations, include the values explicitly. For example:

```text
Calculate CAGR with start_value 50000, end_value 120000, and years 8.
```

---

## 🧮 Financial Calculations

The calculator supports four operations.

### CAGR

```json
{
  "start_value": 50000,
  "end_value": 120000,
  "years": 8
}
```

### Sharpe Ratio

```json
{
  "returns": [0.05, -0.02, 0.08, 0.03],
  "risk_free_rate": 0.04
}
```

### DCF

```json
{
  "fcfs": [100, 120, 140],
  "discount_rate": 0.1,
  "terminal_growth": 0.03
}
```

### Compound Interest

```json
{
  "principal": 100000,
  "annual_rate": 0.08,
  "years": 10,
  "n": 12
}
```

---

## 🧠 Design Decisions

### Why ChromaDB?

Persistent local vector storage with metadata support and straightforward integration with LangChain.

### Why MMR?

Plain similarity search can return highly similar chunks. MMR improves diversity while retaining relevance.

### Why HyDE?

It converts a short user query into a richer hypothetical document representation before retrieval.

### Why RAG Fusion?

Different query formulations can retrieve complementary evidence. Fusion combines these retrieval rankings.

### Why LangGraph?

The project requires conditional routing, tool calls, loops, and persistent state. LangGraph makes those execution paths explicit.

### Why Ollama?

The project can run locally without requiring a paid hosted LLM API.

### Why SQLite?

It provides lightweight persistent checkpointing for conversation threads.

---

## 📁 Project Files

| Path                               | Role                                                                                                                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                           | Main Streamlit application. Creates the chat UI, loads Chroma and Ollama, defines the three agent tools, builds the LangGraph workflow, and stores conversation checkpoints in SQLite.   |
| `ingest.py`                        | Command-line index builder. Loads PDFs from `financial-rag-agent/data/raw`, splits pages into chunks, creates Ollama embeddings, and writes the Chroma database to `vectorstore/chroma`. |
| `requirements.txt`                 | Python dependencies for Streamlit, LangChain, LangGraph, ChromaDB, PDF loading, yfinance, Ollama integration, and notebook support.                                                      |
| `financial-rag-agent/data/raw/`    | Source PDF corpus used by the RAG pipeline. These files are read by `ingest.py` and are not modified by the app.                                                                         |
| `vectorstore/chroma/`              | Generated local Chroma vector database containing the embedded document chunks. It is ignored by Git and must be rebuilt on a new machine.                                               |
| `memory.db`                        | Generated SQLite database used by LangGraph to persist conversation checkpoints. It is ignored by Git.                                                                                   |
| `notebooks/Phase1_RAG.ipynb`       | Notebook for PDF loading, chunking, embedding, Chroma indexing, and basic retrieval experiments.                                                                                         |
| `notebooks/Phase2_Tools.ipynb`     | Notebook for experimenting with market data, news/sentiment, and financial calculator tools.                                                                                             |
| `notebooks/Phase3_Langgraph.ipynb` | Notebook for experimenting with routing, tool calls, graph nodes, and SQLite checkpointing.                                                                                              |
| `.gitignore`                       | Prevents virtual environments, generated databases, vector stores, caches, logs, and local secrets from being committed.                                                                 |
| `README.md`                        | Project documentation, setup instructions, architecture notes, example questions, and file responsibilities.                                                                             |

The notebooks are optional. A normal application run only requires `app.py`, `ingest.py`, the PDF corpus, the installed dependencies, Ollama, and the generated Chroma index.

---

## 🛡️ Reliability Considerations

The agent is designed around several safeguards:

- Financial data is retrieved through tools rather than invented by the model.
- Document answers are instructed to include source/page citations.
- Market information is retrieved dynamically through yfinance.
- Calculations are executed programmatically rather than estimated by the LLM.
- Investment-oriented answers include a risk disclaimer.
- The generation step is instructed to answer from retrieved context.

> **Disclaimer:** This project is an educational/portfolio system and is not financial advice. Market data can change, and retrieved reports may be historical.

---
