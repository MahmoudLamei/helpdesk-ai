# HelpDesk AI — Learning Roadmap

One product, grown phase by phase, that covers the full MLOps lifecycle, then RAG, then agents.
Everything runs locally and free on this machine (8 cores, 23 GB RAM, GTX 1050 4 GB).

## The product

A customer-support assistant for a fictional online shop:

1. **Ticket classifier** (classic ML) — predicts the intent/category of an incoming message.
2. **Knowledge-base Q&A** (RAG) — answers questions from help articles, with citations.
3. **Support agent** (agentic AI) — reads a ticket, classifies it, searches the KB, looks up orders,
   and drafts a reply or escalates to a human.

Each layer reuses the previous one as a *tool*, so the MLOps platform you build in phases 1–6 is
the same one that ships the LLM features in phases 7–10.

**Dataset:** [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(~27k messages, 27 intents, 11 categories, with reference responses). Messages train the classifier;
the responses become the RAG knowledge base.

**Local LLM:** [Ollama](https://ollama.com) with `qwen2.5:3b` or `llama3.2:3b` (fit in 4 GB VRAM, support
tool calling) and `nomic-embed-text` for embeddings.

## How to work through it

- Each phase has a goal, the concepts to learn, tasks, and a **Done when** checklist.
- You write the code. Ask Claude to explain a concept, review your code, or unblock you — but try first.
- Commit at the end of each phase with a tag (`git tag phase-1`). It becomes your portfolio history.
- Budget: roughly 1–2 weeks per phase at a few hours a week.

---

## Part A — MLOps

### Phase 0 · Foundations (scaffolded for you)
**Goal:** a professional Python repo you will build on.
**Learn:** `uv` (envs + lockfiles), src layout, `pyproject.toml`, ruff, pytest, pre-commit, Makefile, git.
**Tasks**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `make install`.
- Run `make lint test`; read every config file and understand each line.
- Create a GitHub repo and push.

**Done when:** `make lint test` passes and pre-commit runs on `git commit`.

### Phase 1 · Data and experiments
**Goal:** a reproducible baseline classifier with tracked experiments.
**Learn:** data versioning, train/val/test splits, stratification, baselines, metrics (macro-F1),
experiment tracking.
**Tasks**
- `src/helpdesk/data/ingest.py`: download the dataset to `data/raw/` (use the `datasets` library).
- Version `data/raw` with **DVC** (local remote first; MinIO in phase 3).
- Notebook EDA: class balance, message lengths, duplicates. Write findings in `docs/eda.md`.
- `src/helpdesk/features/` + `src/helpdesk/train.py`: TF-IDF + LogisticRegression baseline.
- Track params, metrics, confusion matrix, and the model in **MLflow** (`mlflow ui`).
- Try 5+ runs (n-grams, C, a LinearSVC) and compare them in the MLflow UI.

**Done when:** anyone can `git clone && dvc pull && make train` and reproduce your best macro-F1.

### Phase 2 · Pipelines and model registry
**Goal:** training is a pipeline, not a script, and models are promoted by rules.
**Learn:** DAGs, caching, idempotency, config management, model registry, aliases, quality gates.
**Tasks**
- Split into stages (`ingest → validate → featurize → train → evaluate`) in `dvc.yaml`; `dvc repro`.
- Data validation stage (pandera or Great Expectations): schema, nulls, label set.
- Config with pydantic-settings or Hydra — no hard-coded paths or hyperparameters.
- Register models in the MLflow registry; `evaluate` sets the `champion` alias only if the new model
  beats the current champion on the test set.
- Optional: orchestrate the same pipeline with **Prefect** and schedule it.

**Done when:** `dvc repro` rebuilds only what changed, and a worse model can never become champion.

### Phase 3 · Serving
**Goal:** the champion model behind a production-style API.
**Learn:** FastAPI, pydantic schemas, model loading at startup, health/readiness probes, Docker
multi-stage builds, docker compose, load testing.
**Tasks**
- `src/helpdesk/serving/app.py`: `POST /predict`, `GET /health`, loads `models:/helpdesk@champion`.
- Dockerfile (multi-stage, non-root, small image) and `docker-compose.yml` with API + MLflow + MinIO.
- API tests with FastAPI's TestClient; a load test with Locust; record p50/p95 latency.

**Done when:** `docker compose up` gives a working API and you know its latency under load.

### Phase 4 · CI/CD
**Goal:** every push is checked; every merge to main ships an image.
**Learn:** GitHub Actions, caching, test pyramids, building and pushing images, semantic versions.
**Tasks**
- CI: lint, unit tests, train on a small data sample, API tests.
- CD: build and push the image to GHCR on tags.
- Branch protection; PRs only.

**Done when:** a broken test blocks a merge, and `git tag v0.1.0 && git push --tags` publishes an image.

### Phase 5 · Monitoring and retraining
**Goal:** know when the model is degrading and react.
**Learn:** service metrics vs. model metrics, data drift vs. concept drift, prediction logging,
feedback loops.
**Tasks**
- Expose Prometheus metrics from the API (latency, request count, prediction distribution).
- Grafana dashboard in compose.
- Log predictions to SQLite/Postgres; run an **Evidently** drift report on a schedule.
- Simulate drift (e.g. inject new-style messages); trigger the phase-2 pipeline when drift exceeds a threshold.

**Done when:** you can show a dashboard that caught simulated drift and a retrained model that fixed it.

### Phase 6 · Kubernetes
**Goal:** run the stack the way companies do.
**Learn:** kind, Deployments, Services, ConfigMaps/Secrets, probes, rolling updates, HPA, Helm.
**Tasks**
- Local cluster with `kind`; deploy the API with manifests, then convert to a Helm chart.
- Rolling update to a new image tag with zero downtime; roll it back.
- Optional: canary / shadow deployment of a challenger model.

**Done when:** `helm upgrade` rolls out a new model version without failed requests.

---

## Part B — RAG

### Phase 7 · RAG from scratch
**Goal:** understand every piece before using a framework.
**Learn:** chunking, embeddings, vector search, prompt construction, grounding and citations.
**Tasks**
- Build a KB: deduplicate the dataset's responses into help articles per intent (`data/kb/`).
- Embed with `nomic-embed-text` (Ollama); store in **Qdrant** (add to compose).
- `src/helpdesk/rag/`: `retrieve(query, k)` and `answer(query)` with the local LLM; answers cite article ids.
- `POST /ask` endpoint in the API.

**Done when:** `/ask` answers questions with citations, and you can explain each step without notes.

### Phase 8 · RAG evaluation and improvement
**Goal:** measure retrieval and answers, then improve them with evidence.
**Learn:** golden datasets, hit@k / MRR / nDCG, faithfulness, LLM-as-judge (and its limits with small models),
hybrid search, reranking, query rewriting.
**Tasks**
- Build a 100-question golden set (question → expected article).
- Eval script logging retrieval metrics to MLflow; every change is an experiment.
- Try: chunk sizes, BM25 + vector hybrid, a cross-encoder reranker, query rewriting.
- Add the eval to CI as a regression gate.

**Done when:** you have a table of what helped, by how much, and CI fails if retrieval regresses.

---

## Part C — Agents and LLMOps

### Phase 9 · Agent from scratch
**Goal:** see that an agent is a loop: LLM → tool call → result → LLM.
**Learn:** tool/function calling, JSON schemas, the ReAct loop, stop conditions, error handling.
**Tasks**
- Tools: `classify_ticket` (your phase-3 API), `search_kb` (phase 7), `lookup_order` (SQLite with fake
  orders), `escalate_to_human`.
- Write the loop yourself with the Ollama Python client — no framework.
- Test it on 20 tickets; note where the small model fails.

**Done when:** the agent resolves simple tickets end to end and you can read its full trace.

### Phase 10 · Production agents
**Goal:** a reliable, observable, evaluated agent.
**Learn:** LangGraph (state machines, checkpoints), human-in-the-loop, memory, guardrails,
tracing, prompt versioning, agent evals, MCP.
**Tasks**
- Rebuild the agent in **LangGraph**; require human approval before refunds or escalations.
- Trace every run with **MLflow tracing** (or self-hosted Langfuse).
- Version prompts; an agent eval suite (task success, correct tool choice, steps) in CI.
- Guardrails: input validation, PII redaction, output checks.
- Expose your tools as an **MCP server** so any MCP client can use them.

**Done when:** you can change a prompt, run the eval suite, and prove it got better or worse.

---

## Capstone
Write `docs/architecture.md` with a diagram of the whole system and the trade-offs you made, and record a
5-minute demo. That plus the tagged git history is a strong MLOps / LLM engineer portfolio piece.
