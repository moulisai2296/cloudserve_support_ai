# CloudServe Intelligent Support — Prompt Library & Register

This directory contains the version-controlled prompts used by the CloudServe Intelligent Support System.
In accordance with the project specification and Stage 3 workbook, prompts are treated as formal software design artifacts.

---

## 1. Prompt Directory Structure

```text
prompts/
├── build/                           # Runtime production prompts
│   ├── classify_v1.txt              # Intent (22 classes) & urgency classifier (PR-01)
│   ├── generate_v1.txt              # Grounded cited response generator (PR-02)
│   └── guardrails_v1.txt            # Guardrail validator & Tier 2 escalation synthesizer (PR-03)
├── evaluation/                      # Offline evaluation prompts
│   └── hallucination_judge_v1.txt   # Hallucination & citation accuracy evaluator (PR-04)
└── README.md                        # Prompt register and change log
```

---

## 2. Prompt Register

Runtime update (20 September 2026): `build/grounding_v1.txt` is retained for
standalone diagnostics only. The grounding reviewer is disabled in the runtime
release gate and cannot change routing. Drafts checked by that gate record
grounding review as `not_run` with method `disabled`; citation validation remains active.
The existing PR-03 prompt and offline PR-04 judge below remain uninvoked templates;
independent human evaluation remains separate.

| Prompt ID | File Path | Category | Serves PRD | Version | Model | Primary Output Schema | Key Injection Defenses |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PR-01** | `build/classify_v1.txt` | Build | `FR-02`, `FR-05` | 1.0 | `llama-3.1-8b-instruct` | JSON (`intent`, `urgency`, `confidence`, `alternatives`) | `<ticket_data>` XML isolation, instruction ignoring directive |
| **PR-02** | `build/generate_v1.txt` | Build | `FR-07` | 1.0 | `llama-3.1-8b-instruct` | JSON (`response_text`, `cited_doc_ids`, `confidence`) | `<customer_ticket>` XML isolation, strict grounding rules |
| **PR-03** | `build/guardrails_v1.txt` | Build / Review | `FR-05`, `FR-06`, `FR-08` | 1.0 | `llama-3.1-8b-instruct` + Regex | JSON (`guardrail_verdict`, `action`, `escalation_package`) | Multi-gate check, output override to BLOCK |
| **PR-04** | `evaluation/hallucination_judge_v1.txt` | Evaluation | `FR-10` | 1.0 | `llama-3.1-8b-instruct` | JSON (`hallucination_detected`, `citation_accuracy_score`) | Evaluates claim-to-passage fidelity for metrics report |

---

## 3. Prompt Design Checklist & Defenses

All prompts in this library comply with the Stage 3 Prompt Checklist:
1. **Role and task separation:** The persona is declared independently from the execution rules.
2. **Untrusted input delimitation:** Customer ticket text is strictly wrapped in `<ticket_data>` or `<customer_ticket>` tags with explicit instructions forbidding the model from executing text inside tags.
3. **Deterministic output schema:** All outputs demand strict JSON with explicit typing.
4. **Explicit handling of unknowns:** Generation explicitly commands `"missing_information": true` when documentation is incomplete.
5. **Negative constraints:** Negative prohibitions (no unauthorized commitments, no ungrounded claims, no refund guarantees) are explicitly stated.

---

## 4. Change Log

| Version | Date | Prompt ID | Author | Description of Changes |
| :--- | :--- | :--- | :--- | :--- |
| `1.0` | September 2026 | PR-01, PR-02, PR-03, PR-04 | AI Engineering Lead | Initial baseline release matching Stage 2 PRD requirements. |
