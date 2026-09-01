<div align="right">

**English** | [한국어](./README_KO.md)

</div>

# Low-Voltage Motor Maintenance AI Copilot

**Python · Streamlit · RAG · Sentence Transformers · NumPy · Solo Project**

An evidence-grounded maintenance assistant that answers questions from a low-voltage motor operation and maintenance manual and generates checklist / work-order drafts only when supporting evidence is sufficient.

The project runs without an external LLM API. It combines Sentence Transformer retrieval, a lightweight Korean keyword bonus, a rule-based Supervisor, and an evidence-constrained Maintenance Agent to connect **source evidence → task classification → maintenance output**.

> **Portfolio prototype**  
> This is an independent portfolio project and is not affiliated with or endorsed by Hyosung Heavy Industries. The manufacturer's source PDF is not redistributed in the public portfolio edition.

---

## What I Built

- Page-level PDF loading and text cleanup
- Page-boundary-preserving chunking with 600-character chunks / 100-character overlap
- `paraphrase-multilingual-MiniLM-L12-v2` embedding retrieval
- NumPy cosine similarity
- Hybrid ranking with semantic score + limited Korean keyword bonus
- Top-3 source evidence with page-level context
- `insufficient_evidence` gate for weak retrieval
- Supervisor Agent for scope, intent, and requested-output classification
- Synthetic equipment selection through a normal JSON repository
- Evidence-constrained maintenance checklist / work-order drafts
- Early termination for out-of-scope and insufficient-evidence cases
- Streamlit interface and `.txt` work-order draft download
- Unit tests, RAG smoke tests, and evaluation cases

---

## Architecture

```text
User Question
    ↓
Supervisor Agent
    ├─ out_of_scope → Stop
    └─ in_scope
         ↓
Equipment Selection
(maintenance request only)
         ↓
RAG Agent
PDF → page text → chunks → embeddings
         ↓
Hybrid Retrieval
semantic cosine score + keyword bonus
         ↓
Evidence Check
    ├─ insufficient → Stop
    └─ sufficient
         ↓
Maintenance Agent
checklist / work-order draft
         ↓
Orchestrator → Streamlit UI
```

### Supervisor

`src/supervisor_agent.py`

Classifies requests into:

- `manual_query`
- `maintenance_request`
- `out_of_scope`

Out-of-scope questions terminate before retrieval.

### RAG

`src/rag_agent.py`, `src/embedding_index.py`

```text
final_score = semantic_score + keyword_bonus
```

Semantic embedding similarity remains the primary ranking signal. A limited keyword bonus expands selected Korean maintenance terms to improve retrieval for domain-specific wording without replacing semantic search.

Each result retains its source chunk and associated page text so the user can inspect the underlying evidence.

### Maintenance Agent

`src/maintenance_agent.py`

Produces checklist and work-order drafts only from retrieved evidence.

- Does not assert unsupported root causes
- Does not directly order a component replacement
- Stops document generation when evidence is insufficient
- Includes fixed safety-policy checks
- Requires final human review

### Equipment Repository

`data/equipment/motors.json` contains **synthetic demonstration equipment only**, not real facility data. Equipment lookup is a normal JSON query, not vector retrieval, and its provenance is kept separate from manual evidence.

---

## Source Manual

Development used the Korean low-voltage motor operation and maintenance manual published by Hyosung Heavy Industries.

The public portfolio edition does **not** include the manufacturer's PDF.

To run the project locally, download the manual from the official manufacturer source and save it as:

```text
data/manuals/hyosung_motor_manual_ko.pdf
```

Setup details: [`data/manuals/README.md`](data/manuals/README.md)

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Retrieval | Sentence Transformers |
| Similarity | NumPy cosine similarity |
| PDF | pypdf |
| Data | JSON / pandas |
| Testing | pytest |
| Version Control | Git / GitHub |

No external LLM API, LangChain, LangGraph, FAISS, or Chroma is required.

---

## Run Locally

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Add the source manual

Download the Korean low-voltage motor manual from the official manufacturer site and place it at:

```text
data/manuals/hyosung_motor_manual_ko.pdf
```

### 3. Run tests

```bash
.venv\Scripts\python.exe -m pytest tests -v
```

### 4. Run the retrieval smoke test

```bash
.venv\Scripts\python.exe scripts/rag_smoke_test.py
```

### 5. Start Streamlit

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Example Scenarios

**Manual query**

> How should bearing grease be replenished?

Supervisor → RAG retrieval → source-page evidence. The Maintenance Agent is not invoked.

**Maintenance request**

> The motor has severe vibration and metallic noise. Create an inspection checklist.

Supervisor → equipment selection → RAG → evidence check → Maintenance Agent → checklist draft.

**Out of scope**

> What is the weather in Seoul today?

The request terminates at the Supervisor stage.

---

## Problem Solving

### PDF table-of-contents noise

Dot leaders from the PDF table of contents distorted embeddings and caused TOC chunks to outrank the relevant body page for maintenance questions.

I added cleanup logic in `src/pdf_loader.py` to remove repeated dot leaders and added a regression test for the case.

### Korean retrieval quality

Pure semantic retrieval sometimes ranked indirectly related pages above the page containing the direct grease-maintenance procedure.

I added domain-term expansion in `src/keyword_rules.py` and a bounded keyword bonus in `src/embedding_index.py`, improving domain-specific ranking while keeping semantic similarity as the main signal.

### Cache invalidation

Embedding caches need to reflect changes not only to the PDF and chunk settings but also to retrieval logic. An `INDEX_VERSION` value is included in the cache validity rules so retrieval changes can force a clean rebuild.

---

## Repository Structure

```text
app.py
src/
  models.py
  pdf_loader.py
  chunker.py
  keyword_rules.py
  embedding_index.py
  rag_agent.py
  equipment_repository.py
  supervisor_agent.py
  maintenance_agent.py
  orchestrator.py
scripts/
  rag_smoke_test.py
tests/
evaluation/
docs/
data/
  equipment/motors.json
  manuals/README.md
```

---

## Limitations

- The prototype targets one small source manual rather than a broad document corpus.
- Keyword expansion covers only explicitly defined maintenance terminology.
- The similarity threshold is a fixed project setting, not a universally calibrated value.
- The Maintenance Agent uses evidence-grounded deterministic rules/templates rather than a generative LLM.
- Real maintenance decisions, safety procedures, and replacement actions must defer to qualified personnel and manufacturer guidance.

---

## Development Notes

See [`docs/progress_log.md`](docs/progress_log.md) for the staged implementation and debugging record.
