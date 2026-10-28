# HelpDesk AI

A customer-support assistant built step by step to learn MLOps, RAG and agentic AI:
a ticket classifier with a full MLOps lifecycle, a RAG knowledge-base Q&A, and a tool-using agent.

See [ROADMAP.md](ROADMAP.md) for the phases.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
make install
make lint test
```

## Layout

```
src/helpdesk/   package code (data, features, training, serving, rag, agent)
tests/          pytest tests
data/           datasets (versioned with DVC from phase 1)
notebooks/      exploration only; real logic lives in src/
docs/           notes, EDA findings, architecture
```
