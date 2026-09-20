# CloudServe Intelligent Customer Support

An FDE capstone for ticket triage, documentation retrieval, cited response drafting, human escalation, and decision auditing.

## 1. Execute the project and evaluation harness

Run all commands from the repository root. Install **Python 3.14** and **uv** first; these match the project's lockfile and CI configuration.

### Step 1 — Install dependencies

```powershell
uv sync --frozen --python 3.14
```

The commands below use `uv run`, so activating the virtual environment is optional.

### Step 2 — Configure the environment

Create `.env` from the template, preserving any existing configuration:

```powershell
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
```

On macOS/Linux, use `test -f .env || cp .env.example .env` instead.

Edit `.env` privately. Set `OPENROUTER_API_KEY` for model-backed execution. For offline work, leave it empty and use `--offline` in the harness or `"use_llm": false` in API requests. Keep credentials out of source control and recordings.

OpenRouter is the implemented provider. Offline execution uses heuristic classification and extractive generation. Initial dependency and embedding-model downloads can still require internet access.

### Step 3 — Build the documentation index

```powershell
uv run python -m src.retrieve --index
```

This indexes the 29 articles in `data/documentation.json` into Chroma. A populated index is reused. After changing documentation or embedding configuration, rebuild explicitly:

```powershell
uv run python -m src.retrieve --rebuild
```

### Step 4 — Start the application

```powershell
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** and submit a ticket through `POST /tickets`. Keep this terminal running and use another terminal for evaluation.

Example offline request:

```json
{
  "ticket_id": "DEMO-001",
  "channel": "email",
  "subject": "Help setting up MFA",
  "body": "How do I enable multi-factor authentication for my account?",
  "customer_tier": "standard",
  "use_llm": false
}
```

Set `use_llm` to `true` for model-backed processing after configuring the key. Inspect the actual route, draft or escalation packet, citations, and decision ID.

### Step 5 — Run the evaluation harness

The harness runs independently; the API server does not need to be running.

**Quick offline development check — first 10 tickets:**

```powershell
uv run python -m evaluation.harness --input data/development_tickets.json --offline --limit 10 --output evaluation/results/smoke
```

**Full offline development run — 500 tickets:**

```powershell
uv run python -m evaluation.harness --input data/development_tickets.json --offline --output evaluation/results/development
```

**Model-backed development run:**

```powershell
uv run python -m evaluation.harness --input data/development_tickets.json --output evaluation/results/development_llm
```

**Model-backed validation run — all 80 tickets:**

```powershell
uv run python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results
```

With the virtual environment already activated, use:

```powershell
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results
```

Use development data for iteration. The supplied guide reserves validation for a final evaluation; repeated use must be disclosed. A model-enabled run can fall back when the key or provider response is unavailable, so inspect its execution evidence before claiming every ticket used the model successfully.

**Output:** pass only a directory to `--output`. The filenames are fixed:

```text
evaluation/results/
  latest_report.json          Aggregate metrics and run provenance
  latest_ticket_results.json  Per-ticket decisions, sources and findings
```

Each run replaces these files inside its output directory. Use a different directory to preserve each run. Explicit filenames are also supported:

```powershell
uv run python -m evaluation.harness --input data/development_tickets.json --offline --output evaluation/results/dev_run.json --ticket-output evaluation/results/dev_run_tickets.json
```

The console prints each ticket starting, progress, four report sections, and seven KPI comparisons. Incomplete audit reconciliation causes a non-zero exit status. A zero exit status does not certify that every KPI or acceptance requirement passed.

### Step 6 — Run tests

```powershell
uv run pytest tests/ -v
```

For the controlled guardrail-block demonstration:

```powershell
uv run pytest tests/test_guardrails.py::test_guardrails_node_block_and_escalate -v
```

Tests cover components, policy isolation, routing, guardrail suppression, audit reconciliation, batch recovery, API behaviour, and metrics. Retrieval checks may require downloaded embedding weights. Software tests do not establish independent model-answer quality.

## 2. How the system works

Both entry points use `src/graph.py`. Normalisation precedes the graph; retrieval runs before classification.

```text
API / harness -> Normalise -> Retrieve -> Classify -> Route
                                                       |
                         +-----------------------------+-----------+
                         | eligible                                | escalate
                         v                                         |
                      Generate -> Guardrails                       |
                                     | pass or block               |
                                     +------------------+----------+
                                                        v
                                                 Log -> Return
```

| Component | Responsibility |
| --- | --- |
| `src/ingest.py` | Normalises four channel formats |
| `src/retrieve.py` | Chroma search using MiniLM embeddings |
| `src/classify.py` | Predicts 22 intents and three urgency levels with reported confidence |
| `src/route.py` | Applies policies and prepares escalation packets |
| `src/generate.py` | Produces cited drafts or extractive offline responses |
| `src/guardrails.py` | Validates drafts and suppresses blocked responses |
| `src/logging_store.py` | Persists decisions and reconciles batch records |
| `src/api.py` | Ticket-processing and operational endpoints |
| `evaluation/harness.py` | Batch execution, inline metrics, console reporting, and JSON output |

Email, chat, documentation comments, and forums are accepted input formats, not external service connectors. Automatic processing returns a draft; escalation returns a handover payload. Customer delivery, queue assignment, and confirmed resolution are not integrated.

See the [interactive architecture](docs/architecture_diagram.html) for component details and illustrative route traces.

## 3. Configuration

| Variable | Template default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Empty | Model-provider credential |
| `MODEL_NAME` | `meta-llama/llama-3.1-8b-instruct` | Classification and generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings |
| `CHROMA_PATH` | `./storage/chroma` | Vector index |
| `DATABASE_URL` | `sqlite:///./storage/decisions.db` | Audit database |
| `CONFIDENCE_THRESHOLD` | `0.80` | Routing confidence floor |
| `RETRIEVAL_TOP_K` | `3` | Maximum passages |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | `0.40` | Relevance floor |
| `GUARDRAIL_CONFIDENCE_THRESHOLD` | `0.70` | Generation confidence floor |
| `KILL_SWITCH_ACTIVE` | `false` | Forces escalation at routing |

Indexing uses **700-character chunks with 100-character overlap**. Restart the application after editing `.env`; general configuration hot reload is not implemented.

The saved full validation result cited in the report used a retrieval threshold of **0.30**, while the template uses **0.40**. Compare effective settings before interpreting different results. Model-reported confidence is not automatically calibrated probability.

## 4. Routing, guardrails, and audit records

Routing checks apply in order: kill switch, high-risk intents, content exclusions and feature requests, enterprise high urgency, unclear requests, low classification confidence, and missing documentation. Eligible requests proceed to generation.

Content rules cover selected credential exposures, financial disputes/refunds, account or personal-data deletion, and legal/compliance requests. Dataset labels and historical outcomes stay outside runtime inference and are used for scoring and baseline analysis.

Enabled draft checks include selected PII/secret patterns, citation presence and membership in retrieved sources, prompt-injection patterns, unauthorised commitments, generation confidence, and completeness. A block suppresses the draft and creates an escalation.

**The grounding reviewer is disabled.** It does not decide whether to auto-respond or escalate. The guardrail stage records grounding review as `status: not_run` and `method: disabled`; citation validation remains active. Citation-ID validity does not establish support for every factual claim. The reviewer module and prompt remain available for standalone diagnostics.

SQLite records actions, reasons, sources, policy flags, and guardrail findings. Batch reconciliation uses the **run ID and input position**. Previous runs cannot hide missing records, and repeated ticket IDs at different positions require separate decisions.

The harness attempts a logged escalation for an invalid record or processing exception, then continues. Audit-write failures remain visible and prevent complete reconciliation.

## 5. Understand the evaluation report

The console preserves **Volume**, **Business**, **Technical**, and **Governance** sections, followed by seven selected KPIs.

| KPI | Measure | Selected comparison |
| --- | --- | --- |
| KPI-01 | FCR: automatic-response proxy | At least 60% |
| KPI-02 | Escalation rate | At most 30% |
| KPI-03 | P95 processing latency | Under 300 seconds, legacy benchmark |
| KPI-04 | Weighted intent precision | At least 85% |
| KPI-05 | Draft PII/credential scanner detections | Zero |
| KPI-06 | Exact audit reconciliation | 100% |
| KPI-07 | Citation accuracy: retrieved-ID validity proxy | At least 95% |

These operational comparisons have explicit limits:

- Automation is not confirmed customer resolution.
- Processing latency includes audit logging, not customer delivery.
- The framework's **three-second** P95 requirement is separate from the legacy 300-second comparison.
- Weighted precision does not establish that every class meets its target.
- Citation-ID validity and the document-mismatch hallucination heuristic do not replace independent claim review.
- Zero scanner findings do not prove the absence of every kind of private data.

Detailed JSON retains class and urgency metrics, route disagreements, calibration and group diagnostics, configuration, source/dataset hashes, and reconciliation. Outcomes lacking supporting evidence remain separate from the selected operational scorecard.

## 6. API endpoints

| Endpoint | Behaviour |
| --- | --- |
| `POST /tickets` | Returns a draft or escalation payload |
| `GET /health` | Database connectivity and kill-switch state |
| `GET /metrics` | Database-wide operational counts as JSON |
| `GET /admin/kill-switch` | Current process switch state |
| `POST /admin/kill-switch` | Set with `{"active": true}` or `{"active": false}` |
| `GET /docs` | Interactive API documentation |

The kill switch is checked after retrieval and classification. Its API setting is process-local; administrative authentication and a distributed switch are not implemented. The health check does not verify Chroma or the provider, and the metrics endpoint is not Prometheus exposition. The service is a local prototype.

## 7. Data and project files

| Dataset | Contents |
| --- | --- |
| `data/development_tickets.json` | 500 labelled development tickets |
| `data/validation_tickets.json` | 80 labelled validation tickets |
| `data/documentation.json` | 29 knowledge-base articles |
| `data/ground_truth_responses.json` | 200 reference answers; not an active independent judge |

```text
src/                  Application modules
evaluation/           Standalone harness and result artifacts
prompts/              Versioned prompts
tests/                Component and integration checks
data/                 Supplied tickets and documentation
docs/                 Report, script, architecture and workbooks
storage/              Local Chroma and SQLite data; gitignored
.github/workflows/    CI configuration
.env.example          Credential-free configuration template
pyproject.toml        Project and test configuration
uv.lock               Locked dependencies
requirements.txt      Pinned dependency list
```

## 8. Deliverables

- **Project report:** [Markdown](docs/report.md) · [PDF](docs/report.pdf).
- **Video script:** [Markdown](docs/video_script.md) · [PDF](docs/video_script.pdf).
- **Architecture:** [Interactive HTML](docs/architecture_diagram.html).
- **Stage workbooks and effort log:** Word documents in `docs/`.
- **Prompt register:** [prompts/README.md](prompts/README.md).

The report identifies its evidence run and remaining work. The architecture's route traces are illustrative, and its metrics are a dated snapshot rather than live telemetry.

## 9. Troubleshooting

| Symptom | Check |
| --- | --- |
| Unexpected escalation | Per-ticket policy flags, confidence, retrieved scores, and guardrail findings |
| Provider-backed run falls back | Configured key and provider/parsing failures |
| Results differ from an earlier run | Dataset size, mode, effective settings, model, source/prompt hashes, and index |
| First retrieval is slow or fails | Embedding download, index build, and storage access |
| No server after invoking the API module | Use the Uvicorn command in section 1 |
| Earlier results disappear | Directory output replaces fixed filenames; use separate directories |
| Audit reconciliation fails | Missing/extra positions and database write failures |

## 10. Attribution

The project uses LangGraph/LangChain, Chroma, Sentence Transformers, FastAPI, SQLite, scikit-learn, and OpenRouter model access. OpenAI Codex assisted with repository review, implementation and evaluation revisions, and documentation. The author remains responsible for reviewing the code, evidence, and submission declarations.
