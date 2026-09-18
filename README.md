# CloudServe Intelligent Customer Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-green.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Forward Deployed AI Engineering Capstone Project**  
> **Client**: CloudServe Solutions  
> **Mission**: Deliver a production-grade autonomous support triage, grounded technical resolution, and governance-audited routing system for cloud infrastructure and DevOps support.

---

## 1. Executive Summary & Evaluation Targets

CloudServe Solutions provides enterprise cloud infrastructure. This system processes customer support tickets across **4 heterogeneous channels** (Email, Chat, Documentation Comments, and Community Forums), autonomously resolving standard technical inquiries while reliably escalating high-risk, security, or out-of-scope issues to Tier-2 human engineers.

### Core Acceptance Targets

| Dimension | Target Metric | Policy / Operational Rule |
|---|---|---|
| **First Contact Resolution (FCR)** | $\ge 60.0\%$ | Auto-responds only when confidence $\ge 0.80$ and authoritative docs are grounded |
| **Escalation Rate** | $\le 30.0\%$ | Escalates low-confidence, high-risk intents, and policy exclusions |
| **Response Latency** | $< 300\text{s}$ (P95) | Sub-second local retrieval + streaming LLM generation |
| **Intent Classification Precision** | $\ge 85.0\%$ | Calibrated multi-task classifier across 22 ground-truth intents |
| **PII & Secret Leakage** | **0 violations** | *"Block and escalate. Never redact and send."* |
| **Audit Coverage** | **100% (1:1)** | Every decision persisted to SQLite conforming to the 15-field schema |

---

## 2. Architecture & LangGraph Pipeline

The system is orchestrated as a compiled **LangGraph StateGraph** (`src/graph.py`), maintaining a unified `SupportState` across the lifecycle.

```
       [START: raw_ticket]
               │
               ▼
        [retrieve_node] ◄──────── Chroma Vector Store (all-MiniLM-L6-v2)
               │
               ▼
        [classify_node] ◄──────── Llama-3.1-8B via OpenRouter (Few-shot calibrated)
               │
               ▼
         [route_node]   ◄──────── 7-layer deterministic policy router
               │
      { Route Decision? }
      /                 \
  'auto_respond'     'escalate'
    /                     \
[generate_node]            │
   │                       │
[guardrails_node]          │
   │ (Block / Pass)        │
    \                     /
     ▼                   ▼
      [log_decision_node] ◄────── SQLite Audit Store (storage/decisions.db)
               │
             [END]
```

### Component Responsibilities

1. **`src/ingest.py`**: Validates raw payloads across all 4 channels into normalized `NormalizedTicket` models, sanitizing whitespace and extracting customer SLA tiers (`standard`, `business`, `enterprise`).
2. **`src/retrieve.py`**: Chunks 29 knowledge base articles (`RecursiveCharacterTextSplitter`, size 800, overlap 100), embeds with `all-MiniLM-L6-v2` into Chroma, and enforces a relevance similarity threshold (0.40).
3. **`src/classify.py`**: Classifies 22 technical intents and 3 urgency tiers with calibrated confidence scores and candidate alternative intents.
4. **`src/route.py`**: Deterministic 7-tier safety router enforcing Kill-Switch, high-risk intent filtering, exclusion policies, enterprise SLA checks, confidence floors ($\ge 0.80$), and grounding checks. Compiles structured **Tier-2 Escalation Packets**.
5. **`src/generate.py`**: Synthesizes grounded technical resolutions strictly citing retrieved articles in `[DOC-ID]` format.
6. **`src/guardrails.py`**: Pre-release safety gate enforcing PII/secret detection, citation validation against retrieved passages, prompt injection prevention, and unauthorized commitment blocks.
7. **`src/logging_store.py`**: Persists 100% of decisions to SQLite (`storage/decisions.db`) under the 15-field Minimum Record Schema with 1:1 coverage reconciliation.
8. **`src/graph.py`**: Central LangGraph orchestrator stitching all nodes into a compiled `StateGraph` and managing `SupportState`.

> 💡 **Interactive Architecture & Simulation**: Open [`docs/low_level_architecture.html`](docs/low_level_architecture.html) in any browser to inspect the full interactive vector architecture, state mutation lifecycle table, and live step-by-step simulator across 4 real-world ticket scenarios.

---

## 3. Quickstart & Setup (Clean Environment)

### Prerequisites
- Python 3.10 to 3.14
- Recommended: [`uv`](https://docs.astral.sh/uv/) (fast package manager) or standard Python `venv` + `pip`

### Step 1: Clone Repository
```bash
git clone https://github.com/your-org/Capstone_Pack.git
cd Capstone_Pack
```

### Step 2: Set Up Virtual Environment & Dependencies

#### Option A: Using `uv` (Fastest)
```bash
# Install dependencies from uv.lock
uv sync
```

#### Option B: Using standard Python `venv` + `pip`
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```

Edit `.env` and provide your API keys:
```ini
# Model Provider (OpenRouter or Groq)
OPENROUTER_API_KEY=sk-or-v1-your-real-key-here
MODEL_NAME=meta-llama/llama-3.1-8b-instruct
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Storage Locations
CHROMA_PATH=./storage/chroma
DATABASE_URL=sqlite:///./storage/decisions.db

# Routing Thresholds & Safety
CONFIDENCE_THRESHOLD=0.80
RETRIEVAL_SIMILARITY_THRESHOLD=0.40
KILL_SWITCH_ACTIVE=false
```

---

## 4. Running the System

### Step 4.1: Build / Verify the Knowledge Base Index
Indexes the 29 official documentation articles into Chroma vector storage:
```bash
python -m src.retrieve --index
```
*(Optionally force a clean rebuild with `python -m src.retrieve --rebuild`)*

---

### Step 4.2: Run the Unattended Evaluation Gate (The Gate — AC-A9)
Runs the standalone batch evaluation harness in-memory across the 120-ticket validation dataset, prints an ASCII scorecard, and writes the JSON report:
```bash
# Run over validation dataset
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/validation_report.json

# Quick smoke-test (first 10 tickets)
python -m evaluation.harness --input data/validation_tickets.json --limit 10 --output evaluation/results/smoke_test.json
```

---

### Step 4.3: Run the Automated Test Suite (53 Tests — AC-A12)
Executes the comprehensive pytest suite covering all components, edge cases, safety guardrails, and audit logging:
```bash
# Using uv:
uv run pytest

# Or with active venv:
pytest -v
```

---

### Step 4.4: Start the FastAPI Production Service
Launches the REST API server exposing the triage pipeline, metrics, and administration:
```bash
# Run API server
python -m src.api
# Or via uvicorn directly:
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints:
- `POST /tickets` — Submit a customer ticket for automated triage and resolution.
- `GET /health` — Service readiness, Chroma vector store check, and database connection.
- `GET /metrics` — Live operational statistics (total processed, auto-respond count, escalation rate).
- `POST /admin/kill-switch` — Administrative override to toggle the system-wide kill-switch.
- `GET /docs` — Interactive OpenAPI / Swagger documentation (`http://localhost:8000/docs`).

---

## 5. Governance, Safety & Compliance

### Deterministic Policy Routing (`src/route.py`)
Incoming requests undergo 7 mandatory checks in strict order:
1. **Emergency Kill-Switch**: If active, routes 100% of tickets to human escalation.
2. **High-Risk Intent Filter**: Keywords or intents involving credential exposure, data loss, outage incidents, or legal disputes trigger immediate escalation.
3. **Exclusion Policy**: Account deletion, billing refunds, and custom contract terms cannot be automated.
4. **Enterprise SLA**: Critical-urgency enterprise tickets generate Tier-2 escalation packets immediately.
5. **Confidence Floor**: Any classification with confidence $< 0.80$ is escalated (`low_confidence`).
6. **Grounding Requirement**: Inquiries with no relevant documentation matches are escalated (`no_relevant_docs`).
7. **Safe Auto-Response**: All conditions met $\to$ routed to resolution generation.

### Safety Guardrails (`src/guardrails.py`)
All drafted resolutions are checked before customer release:
- **PII / Secret Leakage**: Detects leaked API keys, tokens, SSH keys, passwords, and sensitive emails.
- **Citation Authenticity**: Regex extracts all `[DOC-ID]` references and verifies them against retrieved passage IDs.
- **Prompt Injection**: Scans for jailbreaks and unauthorized instruction overrides.
- **Golden Rule**: *"Block and escalate. Never redact and send."* If tripped, the response is wiped to `None` and the ticket is escalated.

### SQLite Decision Audit Store (`src/logging_store.py`)
Every decision (auto-respond, escalate, or block) is recorded in `storage/decisions.db` with:
- `decision_id`, `timestamp`, `ticket_id`, `channel`, `customer_tier`
- `intent`, `confidence`, `urgency`
- `action_taken` (`auto_respond` | `escalate` | `block`), `reason`
- `sources_used` (exact document IDs and similarity scores)
- `escalation_packet`, `guardrail_results`, `policy_flags`

**Reconciliation**: The evaluation harness calls `reconcile_decisions_with_tickets()` to assert that 100% of evaluated tickets were persisted to the database.

---

## 6. Repository Layout

```text
├── README.md                      # Setup and operational instructions (AC-A1)
├── pyproject.toml                 # Project metadata and test configuration
├── requirements.txt               # Pinned dependency manifest
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core pipeline implementation
│   ├── ingest.py                  # Multi-channel normalization (AC-A2 / B-02)
│   ├── retrieve.py                # Chroma vector store & dense search (AC-A4 / B-03, B-04)
│   ├── classify.py                # Intent & urgency classification (AC-A3 / B-06)
│   ├── route.py                   # Deterministic policy router (AC-A5 / B-07)
│   ├── generate.py                # Grounded technical resolution (AC-A6 / B-08)
│   ├── guardrails.py              # Pre-release safety gate (AC-A7 / B-09)
│   ├── logging_store.py           # SQLite decision audit store (AC-A8 / B-10)
│   ├── graph.py                   # Central LangGraph StateGraph & SupportState
│   └── api.py                     # FastAPI REST service & admin controls
│
├── evaluation/                    # Automated evaluation suite
│   ├── harness.py                 # Standalone CLI batch runner (AC-A9, AC-A10 / B-05, B-11)
│   └── results/                   # JSON evaluation scorecards & reports
│
├── data/                          # Ground-truth datasets
│   ├── documentation.json         # 29 official knowledge base articles
│   ├── development_tickets.json   # 500 development training/calibration tickets
│   └── validation_tickets.json    # 120 unattended validation gate tickets
│
├── prompts/                       # Prompt template registry
│   ├── build/                     # Runtime prompts (classify_v1.txt, generate_v1.txt)
│   ├── evaluation/                # Evaluation judge prompts
│   └── README.md                  # Prompt versioning register
│
├── tests/                         # Pytest test suite (53 passing tests)
│   ├── test_ingest.py             # Channel normalization & schema validation
│   ├── test_retrieve.py           # Embedding, indexing, and similarity search
│   ├── test_classify.py           # Classification & confidence calibration
│   ├── test_route.py              # Policy routing & escalation packet building
│   ├── test_generate.py           # Citation extraction & response grounding
│   ├── test_guardrails.py         # Safety rules, PII detection, and blocking
│   ├── test_logging_store.py      # SQLite logging & reconciliation
│   ├── test_graph.py              # LangGraph compilation & end-to-end execution
│   ├── test_api.py                # FastAPI endpoints & HTTP integration
│   └── test_harness.py            # Evaluation metrics computation
│
├── docs/                          # Documentation & visual assets
│   ├── low_level_architecture.html # Interactive Architecture, LangGraph, & Simulator Blueprint
│   └── ...                        # Architecture and reference documents
│
└── storage/                       # Local runtime persistence (gitignored)
    ├── chroma/                    # Chroma persistent vector database
    └── decisions.db               # SQLite governance audit database
```

---

## 7. License & Credits

Built for the **CloudServe Solutions Forward Deployed AI Engineering Capstone**.  
Licensed under the [MIT License](LICENSE).
