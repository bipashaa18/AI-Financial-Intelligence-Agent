import streamlit as st
import sqlite3
import json
import numpy as np
import yfinance as yf
import torch
import operator
import uuid

torch.classes.__path__ = []

from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Agent",
    page_icon="💰",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_1"
if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════════════════
# LOAD EVERYTHING ONCE
# @st.cache_resource = runs once, reused on every interaction
# Without this, the agent would reload on every message
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_app():

    # ── PHASE 1: ChromaDB + Retriever ──────────────────────────
    # Connects to the FAISS index built in Phase1_RAG.ipynb
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chroma_db  = Chroma(
        collection_name="financial_docs",
        embedding_function=embeddings,
        persist_directory="vectorstore/chroma",
    )
    base_retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
    )

    # ── LLM ────────────────────────────────────────────────────
    llm = ChatOllama(model="llama3.2", temperature=0)

    # ── PHASE 2: Tools ─────────────────────────────────────────
    # Tool 1: RAG — searches your 33 PDFs (Phase 1)
    @tool(response_format="content_and_artifact")
    def search_financial_docs(query: str):
        """
        Search internal financial documents (RBI, IMF, Gold, Oil reports).
        Use for: policy data, historical stats, institutional forecasts,
        inflation data, GDP, monetary policy, risk factors from reports.
        Do NOT use for live prices or today's news.
        """
        docs = base_retriever.invoke(query)[:5]
        if not docs:
            return "No relevant documents found.", []
        context = "\n\n".join(
            f"[{d.metadata.get('source','?')}, page {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )
        return f"## Document Results\n\n{context}", docs

    # Tool 2: Live market data — yfinance (no API key, instant)
    @tool(response_format="content_and_artifact")
    def get_market_data(ticker: str):
        """
        Get live stock or commodity market data from Yahoo Finance.
        Use for: current price, P/E ratio, market cap, analyst rating,
        52-week high/low, dividend yield.
        Ticker examples: AAPL, RELIANCE.NS, TCS.NS, GC=F (gold),
        CL=F (oil), ^NSEI (Nifty 50), ^BSESN (Sensex)
        """
        try:
            info = yf.Ticker(ticker).info
            data = {
                "name":           info.get("longName", "N/A"),
                "ticker":         ticker,
                "price":          info.get("currentPrice") or info.get("regularMarketPrice"),
                "currency":       info.get("currency", "USD"),
                "pe_ratio":       info.get("trailingPE"),
                "market_cap":     info.get("marketCap"),
                "52w_high":       info.get("fiftyTwoWeekHigh"),
                "52w_low":        info.get("fiftyTwoWeekLow"),
                "analyst_rating": info.get("recommendationKey", "N/A"),
                "target_price":   info.get("targetMeanPrice"),
                "beta":           info.get("beta"),
            }
            content = f"## Market Data: {ticker}\n\n{json.dumps(data, indent=2, default=str)}"
            return content, data
        except Exception as e:
            return f"Error fetching {ticker}: {e}", {}

    # Tool 3: Financial calculator — pure Python, instant
    @tool(response_format="content_and_artifact")
    def financial_calculator(calc_type: str, parameters: str):
        """
        Run financial calculations. No external API needed.
        calc_type options and parameters:

        cagr:     {"start_value":50000, "end_value":120000, "years":8}
        sharpe:   {"returns":[0.05,-0.02,0.08,0.03], "risk_free_rate":0.04}
        dcf:      {"fcfs":[100,120,140], "discount_rate":0.10, "terminal_growth":0.03}
        compound: {"principal":100000, "annual_rate":0.08, "years":10, "n":12}

        Use for: CAGR calculation, Sharpe ratio, DCF valuation,
        compound interest, any numerical financial computation.
        """
        try:
            p = json.loads(parameters)

            if calc_type == "cagr":
                cagr   = (p["end_value"] / p["start_value"]) ** (1/p["years"]) - 1
                result = {
                    "cagr_pct":         round(cagr * 100, 2),
                    "total_return_pct": round((p["end_value"]/p["start_value"] - 1) * 100, 2),
                }

            elif calc_type == "sharpe":
                r      = np.array(p["returns"])
                sr     = (np.mean(r) - p.get("risk_free_rate", 0.04)) / np.std(r, ddof=1)
                result = {
                    "sharpe_ratio":   round(float(sr), 4),
                    "interpretation": (
                        "Excellent" if sr > 1 else
                        "Good"      if sr > 0.5 else
                        "Poor"
                    ),
                }

            elif calc_type == "dcf":
                r, g   = p.get("discount_rate", 0.10), p.get("terminal_growth", 0.03)
                pvs    = [f/(1+r)**t for t, f in enumerate(p["fcfs"], 1)]
                tv     = p["fcfs"][-1] * (1+g) / (r-g)
                result = {
                    "pv_cashflows":    round(sum(pvs), 2),
                    "terminal_value":  round(tv, 2),
                    "intrinsic_value": round(sum(pvs) + tv/(1+r)**len(p["fcfs"]), 2),
                }

            elif calc_type == "compound_interest":
                P, r, n, t = p["principal"], p["annual_rate"], p.get("n", 12), p["years"]
                A          = P * (1 + r/n) ** (n*t)
                result     = {
                    "future_value":    round(A, 2),
                    "interest_earned": round(A - P, 2),
                }

            else:
                return f"Unknown calc_type: {calc_type}. Use: cagr, sharpe, dcf, compound_interest", {}

            content = f"## {calc_type.upper()} Result\n\n{json.dumps(result, indent=2)}"
            return content, result

        except Exception as e:
            return f"Calculation error: {e}", {}

    ALL_TOOLS = [
        search_financial_docs,
        get_market_data,
        financial_calculator,
    ]

    # ── PHASE 3: LangGraph State ────────────────────────────────
    class FinancialAgentState(MessagesState):
        query:           str
        retrieved_docs:  Annotated[list[Document], operator.add]
        context:         Annotated[str, operator.add]
        generation:      str
        needs_retrieval: bool

    # Rule-based router — fast, reliable, no LLM call needed
    KEYWORDS = [
        "price", "stock", "ticker", "market", "pe ratio", "p/e",
        "dividend", "analyst", "nifty", "sensex", "gold", "oil",
        "crude", "aapl", "reliance", "tcs", "hdfc", "infy",
        "calculate", "cagr", "sharpe", "dcf", "compound",
        "report", "rbi", "imf", "forecast", "outlook", "policy",
        "inflation", "gdp", "monetary", "fiscal", "risk",
        "what is", "what are", "explain", "define", "how does",
        "investment", "return", "portfolio", "fund", "bond",
        "equity", "commodity", "currency", "rate", "interest",
    ]
    TICKERS = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
        "RELIANCE", "TCS", "HDFC", "INFY", "WIPRO",
        "GC=F", "CL=F", "^NSEI", "^BSESN", "USDINR=X",
    ]

    def route_question(state: FinancialAgentState) -> dict:
        query = state["query"].lower()
        needs = (
            any(kw in query for kw in KEYWORDS)
            or any(t in state["query"] for t in TICKERS)
        )
        return {"needs_retrieval": needs}

    def route_edge(state: FinancialAgentState) -> Literal["agent", "generate"]:
        return "agent" if state["needs_retrieval"] else "generate"

    # System prompt — tells agent when to use each tool
    SYSTEM_PROMPT = """You are an expert AI Financial Analyst.

You have 3 tools. Use them based on the question:

1. search_financial_docs
   → Use for: RBI/IMF/Gold/Oil reports, policy documents,
     inflation data, GDP, monetary policy, historical stats.
   → Always cite: [source file, page number]

2. get_market_data
   → Use for: current stock price, P/E ratio, market cap,
     analyst rating, 52-week range, beta.
   → Provide the correct ticker symbol.
   → Examples: AAPL, RELIANCE.NS, GC=F, CL=F, ^NSEI

3. financial_calculator
   → Use for: CAGR, Sharpe ratio, DCF, compound interest.
   → User must provide the numbers.

RULES:
- Always use tools. Never guess financial data.
- For complex questions, call multiple tools in sequence.
- Cite document sources: [filename, page]
- Add a brief risk disclaimer for investment advice.
- Be concise and structured in your answers.
"""

    GEN_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a financial analyst. Answer using ONLY the context provided. "
         "If context is empty, answer from general knowledge. "
         "Always cite sources. Add risk disclaimer for investment advice. "
         "Be clear and structured."),
        ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    ])

    gen_chain            = GEN_PROMPT | llm | StrOutputParser()
    agent_llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ── PHASE 3: Graph Nodes ────────────────────────────────────
    MAX_TOOL_CALLS = 3

    def agent(state: FinancialAgentState) -> dict:
        msgs = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
            *state["messages"],
        ]
        return {"messages": [agent_llm_with_tools.invoke(msgs)]}

    def should_continue(state: FinancialAgentState) -> Literal["tools", "generate"]:
        last = state["messages"][-1]
        tool_call_count = sum(
            len(message.tool_calls)
            for message in state["messages"]
            if isinstance(message, AIMessage) and message.tool_calls
        )
        if hasattr(last, "tool_calls") and last.tool_calls and tool_call_count < MAX_TOOL_CALLS:
            return "tools"
        return "generate"

    def collect_tool_output(state: FinancialAgentState) -> dict:
        context = "\n\n".join(
            msg.content
            for msg in state["messages"]
            if isinstance(msg, ToolMessage)
        )
        return {"context": context or ""}

    def generate(state: FinancialAgentState) -> dict:
        return {"generation": gen_chain.invoke({
            "context": state.get("context", ""),
            "query":   state["query"],
        })}

    # ── PHASE 3: Build Graph ────────────────────────────────────
    conn   = sqlite3.connect("memory.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(FinancialAgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("collect_tool_output", collect_tool_output)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_edge)
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "collect_tool_output")
    graph.add_edge("collect_tool_output", "agent")
    graph.add_edge("generate", END)

    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("💰 AI Financial Intelligence Agent")
st.markdown(
    """
    Ask me about **stocks, commodities, financial calculations**, or **financial reports** 
    (RBI, IMF, Gold, Oil).
    
    I can search documents, fetch live market data, and perform financial calculations.
    """
)

# Load the agent once
agent = load_app()

# Display chat history
for msg in st.session_state.messages:
    if msg.get("role") == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# User input
user_input = st.chat_input("Ask me anything about finances...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                result = agent.invoke(
                    {
                        "query": user_input,
                        "messages": [],
                    },
                    config,
                )
                
                response = result.get("generation", "No response generated.")
                st.write(response)
                
                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})