# 💰 AI Financial Intelligence Agent

An agentic financial RAG system that combines **advanced document retrieval, live market data, financial calculations, and conversational memory** in a single Streamlit application.

The project is designed as a portfolio/placement-ready demonstration of **RAG, agent orchestration, tool calling, vector databases, local LLMs, and production-style UI integration**.

---

## 🚀 What It Does

The agent can:

- Search **33 institutional financial PDFs** using an advanced RAG pipeline.
- Improve retrieval with **HyDE (Hypothetical Document Embeddings)**.
- Expand queries with **RAG Fusion + Reciprocal Rank Fusion (RRF)**.
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
        │ HyDE           │  │ yfinance       │  │ CAGR            │
        │ RAG Fusion     │  │ Live prices    │  │ Sharpe          │
        │ MMR Retrieval  │  │ P/E, beta etc. │  │ DCF             │
        │ ChromaDB       │  │                │  │ Compound Interest│
        └───────┬────────┘  └────────────────┘  └─────────────────┘
                │
                ▼
       ┌─────────────────────┐
       │ 33 Financial PDFs    │
       │ RBI / IMF / OECD /   │
       │ Gold / Oil / Reports │
       └─────────────────────┘

                  ┌─────────────────────┐
                  │ SQLite Checkpointer │
                  │ Conversation Memory │
                  └─────────────────────┘
```

---

## 🔎 Advanced RAG Pipeline

The document-search tool uses multiple retrieval strategies instead of relying on a single vector search.

### 1. HyDE

The user's question is first converted into a hypothetical financial-report passage.

```text
Question
   ↓
LLM generates hypothetical report passage
   ↓
Embed hypothetical passage
   ↓
ChromaDB retrieval
```

This can improve semantic matching when the wording of the question differs from the wording used in the source documents.

### 2. RAG Fusion

The original question is expanded into multiple plain-English search queries.

```text
Original question
       ↓
3 related search queries
       ↓
3 independent retrievals
       ↓
Reciprocal Rank Fusion
       ↓
Top-ranked documents
```

### 3. MMR Retrieval

ChromaDB uses **Maximal Marginal Relevance (MMR)** to balance relevance with diversity and reduce redundant chunks.

### 4. Deduplication

HyDE and Fusion results are combined and deduplicated before the final top documents are passed to the agent.

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

| Tool | Purpose |
|---|---|
| `search_financial_docs` | Search the internal financial document collection |
| `get_market_data` | Retrieve current market information through yfinance |
| `financial_calculator` | Perform financial calculations |

---

## 📚 Data

The RAG corpus contains **33 institutional financial PDFs**, including material related to:

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

Current project notes indicate approximately **32,300 indexed chunks**.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama `llama3.2` |
| Embeddings | Ollama `nomic-embed-text` |
| RAG | LangChain |
| Agent orchestration | LangGraph |
| Vector database | ChromaDB |
| Retrieval | MMR + HyDE + RAG Fusion |
| Market data | yfinance |
| Memory | SQLite + LangGraph checkpointing |
| UI | Streamlit |
| Language | Python |


## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/financial-rag-agent.git
cd financial-rag-agent
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

Make sure Ollama is installed and running:

```bash
ollama serve
```

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Prepare the vector database

The ChromaDB directory must contain the indexed financial documents before launching the application.

```text
vectorstore/chroma/
```

### 6. Start Streamlit

```bash
streamlit run app.py
```

The application should open in your browser.

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
  "discount_rate": 0.10,
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
