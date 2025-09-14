# Intelligent Content Analyzer — Architecture

This document explains the system architecture, data flow, and how the repository maps to the running services. It uses a C4 model to move from a high-level overview to detailed components.

- Source of truth for app wiring:
  - API Gateway: [services/api_gateway/app/main.py](services/api_gateway/app/main.py)
  - Routers:
    - Upload: [`services.api_gateway.app.routers.upload`](services/api_gateway/app/routers/upload.py)
    - QA: [`services.api_gateway.app.routers.qa`](services/api_gateway/app/routers/qa.py)
    - Summary: [`services.api_gateway.app.routers.summary`](services/api_gateway/app/routers/summary.py)
  - Retrieval: [services/retrieval/app/main.py](services/retrieval/app/main.py), hybrid: [services/retrieval/app/hybrid.py](services/retrieval/app/hybrid.py), FAISS: [services/retrieval/app/faiss_store.py](services/retrieval/app/faiss_store.py), reranker: [services/retrieval/app/rerank.py](services/retrieval/app/rerank.py)
  - Ingestion: [services/ingest/app/main.py](services/ingest/app/main.py), readers: [services/ingest/app/readers.py](services/ingest/app/readers.py), chunkers: [services/ingest/app/chunkers.py](services/ingest/app/chunkers.py)
  - LLM Generation: [services/llm_generate/app/main.py](services/llm_generate/app/main.py), prompts: [services/llm_generate/app/prompts.py](services/llm_generate/app/prompts.py)
  - Embeddings: [services/embeddings/app/main.py](services/embeddings/app/main.py), helpers: [services/embeddings/app/embeddings.py](services/embeddings/app/embeddings.py)
  - Evaluation: [services/evaluation/app/main.py](services/evaluation/app/main.py), confidence: [services/evaluation/app/confidence.py](services/evaluation/app/confidence.py), metrics: [services/evaluation/app/metrics.py](services/evaluation/app/metrics.py)
  - Shared libs: Models [shared/models.py](shared/models.py), Cache [shared/cache.py](shared/cache.py), Settings [shared/settings.py](shared/settings.py), Tracing [shared/tracing.py](shared/tracing.py)
  - Orchestration: [infra/docker-compose.yml](infra/docker-compose.yml), env: [.env.example](.env.example)
  - UI: [ui/app.py](ui/app.py)

Notes
- Dual run modes:
  - Local single-process: API Gateway imports other services in-proc (no servers).
  - HTTP microservices: When upstream URLs are set, API Gateway calls services over HTTP:
    - INGEST_URL, RETRIEVAL_URL, LLM_GENERATE_URL, EVALUATION_URL
- Retrieval exposes status/debug endpoints used in ops: [`/status`](services/retrieval/app/main.py), [`/chunks_by_doc`](services/retrieval/app/main.py), [`/debug/storage`](services/retrieval/app/main.py).

## Repository structure

- services/
  - api_gateway/ … FastAPI, routes, orchestration (HTTP or in-proc via toggles)
  - ingest/ … file parsing and chunking
  - retrieval/ … BM25 + dense hybrid, FAISS, reranker
  - embeddings/ … CPU-friendly embeddings abstraction (offline deterministic supported)
  - llm_generate/ … Gemini-based answer/summarize, router prompt, strict JSON parsing + citation heuristics
  - evaluation/ … heuristic metrics + optional LLM-as-judge, confidence blending
- shared/ … models, settings, cache, tracing
- infra/ … docker-compose and environment
- documentation/ … this documentation
- tests/ … unit tests
- ui/ … Streamlit client calling only the API Gateway

## C4 — Level 1: System context

```mermaid
%%{init: {"themeVariables": {"fontSize": "24px"}, "flowchart": {"htmlLabels": true}}}%%
C4Context
    title Intelligent Content Analyzer — System Context
    Person(user, "Student/User", "Uploads documents, asks questions, requests summaries")
    System_Ext(ui, "Streamlit UI", "Calls only API Gateway")
    System_Boundary(ica, "Intelligent Content Analyzer") {
      System(api, "API Gateway", "FastAPI")
      System(ing, "Ingest Service", "Parsers + chunkers")
      System(ret, "Retrieval Service", "BM25 + Dense + FAISS")
      System(embed, "Embeddings Service", "Embeddings API or offline")
      System(llm, "LLM Generation Service", "Gemini QA/Summarize")
      System(eval, "Evaluation Service", "Heuristics + LLM-as-judge")
      System_Ext(redis, "Redis (optional)", "Cache (exact/semantic/index-versioned)")
      System_Ext(faiss, "FAISS index", "Persistent vector store")
      System_Ext(langfuse, "Langfuse (optional)", "Tracing/observability")
    }
    Rel(user, ui, "Browser")
    Rel(ui, api, "HTTP: /upload_document, /ask_question, /document_summary")
    Rel(api, ing, "HTTP (when INGEST_URL) or in-proc")
    Rel(api, ret, "HTTP (when RETRIEVAL_URL) or in-proc")
    Rel(api, llm, "HTTP (when LLM_GENERATE_URL) or in-proc")
    Rel(api, eval, "HTTP (when EVALUATION_URL) or in-proc")
    Rel(ret, embed, "HTTP (when EMBEDDINGS_URL) or in-proc")
    Rel(api, redis, "Cache get/set")
    Rel(ret, faiss, "Upsert/search vectors")
    Rel(api, langfuse, "Tracing spans/events")
```

## C4 — Level 2: Containers

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"htmlLabels": true}}}%%
C4Container
    title Intelligent Content Analyzer — Containers
    Person(user, "Student/User")
    System_Ext(ui, "Streamlit UI", "ui/app.py")
    System_Boundary(ica, "ICA") {
      Container(api, "API Gateway", "FastAPI", "Routes: upload, QA, summary; Orchestrates retrieval→(rerank)→LLM→eval→confidence; Caching")
      Container(ing, "Ingest", "FastAPI", "Multipart parse, readers, chunkers; returns chunks (JSON/NDJSON)")
      Container(retr, "Retrieval", "FastAPI", "BM25 + dense blend (/search), FAISS persistence; /status, /chunks_by_doc")
      Container(emb, "Embeddings", "FastAPI", "HTTP embeddings or offline deterministic")
      Container(llm, "LLM Generate", "FastAPI", "Gemini QA/Summarize with router prompt; strict JSON parsing + citation heuristics")
      Container(ev, "Evaluation", "FastAPI", "Heuristic metrics + optional LLM-as-judge; confidence module")
      ContainerDb(redis, "Redis (optional)", "Cache", "Exact + semantic + fingerprint; index-version invalidation")
      ContainerDb(faiss, "FAISS", "Index", "Persistent vectors + doc map")
      Container_Ext(langfuse, "Langfuse (optional)", "Tracing", "Spans/events")
    }
    Rel(user, ui, "HTTP(S)")
    Rel(ui, api, "HTTP")
    Rel(api, ing, "HTTP or in-proc (toggle by *_URL env)")
    Rel(api, retr, "HTTP or in-proc (toggle by *_URL env)")
    Rel(api, llm, "HTTP or in-proc (toggle by *_URL env)")
    Rel(api, ev, "HTTP or in-proc (toggle by *_URL env)")
    Rel(retr, emb, "HTTP or in-proc (toggle by *_URL env)")
    Rel(api, redis, "Cache exact/semantic/versioned")
    Rel(retr, faiss, "Upsert/search/remove")
    Rel(api, langfuse, "Spans/events")
```

## C4 — Level 3: Components (API Gateway)

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"htmlLabels": true}}}%%
flowchart TB
  subgraph API_Gateway["API Gateway (services/api_gateway/app)"]
    upload["Upload Router (/upload_document) — [`upload.py`](services/api_gateway/app/routers/upload.py)"]
    qa["QA Router (/ask_question) — [`qa.py`](services/api_gateway/app/routers/qa.py)"]
    summary["Summary Router (/document_summary) — [`summary.py`](services/api_gateway/app/routers/summary.py)"]
    cache_client["Cache Client — [`shared/cache.py`](shared/cache.py)"]
    tracing["Tracing — [`shared/tracing.py`](shared/tracing.py)"]
    settings["Settings — [`shared/settings.py`](shared/settings.py)"]
    models["Models — [`shared/models.py`](shared/models.py)"]
  end

  Retrieval["Retrieval Service — [`services/retrieval/app`](services/retrieval/app/main.py)"]
  LLM_Generate["LLM Generate — [`services/llm_generate/app`](services/llm_generate/app/main.py)"]
  Evaluation["Evaluation — [`services/evaluation/app`](services/evaluation/app/main.py)"]
  Embeddings["Embeddings — [`services/embeddings/app`](services/embeddings/app/main.py)"]
  FAISS[("FAISS Index + Doc Map — [`faiss_store.py`](services/retrieval/app/faiss_store.py)")]
  Cache[("Redis Cache (optional) — [`shared/cache.py`](shared/cache.py)")]
  Langfuse["Langfuse (optional) — [`shared/tracing.py`](shared/tracing.py)"]

  %% Internal relationships
  qa --> cache_client
  qa --> tracing
  upload --> tracing
  summary --> tracing

  %% API Gateway to services
  upload --> Retrieval
  qa --> Retrieval
  qa --> LLM_Generate
  qa --> Evaluation
  summary --> Retrieval
  summary --> LLM_Generate

  %% Retrieval dependencies
  Retrieval --> Embeddings
  Retrieval --> FAISS

  %% Cache usage
  qa <---> Cache
  summary <---> Cache
  upload --> Cache

  %% Observability
  qa -. spans/events .-> Langfuse
  Retrieval -. spans/events .-> Langfuse
  LLM_Generate -. spans/events .-> Langfuse
  Evaluation -. spans/events .-> Langfuse
```

## Updated data flow diagrams

### End-to-end flow (upload, ask, summarize)

```mermaid
%%{init: {"themeVariables": {"fontSize": "20px"}, "flowchart": {"htmlLabels": true}}}%%
flowchart LR
  %% Ingestion & Indexing
  subgraph Ingestion_and_Indexing
    UPLOAD["POST /upload_document (Gateway)"]
    P["Readers: PDF/DOCX/PPTX/HTML/MD/Image<br/>(OCR/caption hooks) — [`readers.py`](services/ingest/app/readers.py)"]
    C["Chunkers: section-aware (pages/headings/tables) — [`chunkers.py`](services/ingest/app/chunkers.py)"]
    SAN["Sanitize meta for index (avoid 422) — [`upload.py`](services/api_gateway/app/routers/upload.py)"]
    IDX["Retrieval /index — [`main.py`](services/retrieval/app/main.py)"]
    E["Embeddings — [`embeddings.py`](services/embeddings/app/embeddings.py)"]
    VDB[("FAISS index + doc_map — [`faiss_store.py`](services/retrieval/app/faiss_store.py)")]
    UPLOAD --> P --> C --> SAN --> IDX --> E --> VDB
  end
  client1[(Client/UI)]
  UPLOAD -->|doc_id| client1

  %% Question Answering
  subgraph Question_Answering
    ASK["POST /ask_question (Gateway) — [`qa.py`](services/api_gateway/app/routers/qa.py)"]
    R["Hybrid Retrieval: BM25 + Dense (+RRF union) — [`hybrid.py`](services/retrieval/app/hybrid.py)"]
    RR["Re-ranker (optional) — [`rerank.py`](services/retrieval/app/rerank.py)"]
    THR{"Threshold gate / refine / translate"}
    G["LLM Generate (Gemini) — [`llm_generate/main.py`](services/llm_generate/app/main.py)"]
    EV["Evaluate + Confidence — [`evaluation/main.py`](services/evaluation/app/main.py), [`confidence.py`](services/evaluation/app/confidence.py)"]
    A[["Answer + citations + confidence"]]
    ASK --> R --> RR --> THR
    THR -- "low" --> R
    THR -- "ok" --> G
    R --> G
    G --> EV --> A
  end
  A --> client1

  %% Summarization
  subgraph Summarization
    SUM["GET /document_summary (Gateway) — [`summary.py`](services/api_gateway/app/routers/summary.py)"]
    CH["Retrieval /chunks_by_doc — [`main.py`](services/retrieval/app/main.py)"]
    SG["LLM Summarize — [`llm_generate/main.py`](services/llm_generate/app/main.py)"]
    SUM --> CH --> SG
  end

  %% Caching
  subgraph Caching
    CACHE[("Redis (exact/semantic/index-versioned) — [`shared/cache.py`](shared/cache.py)")]
  end
  ASK -. "exact/semantic lookup" .-> CACHE
  A -. "set versioned keys + semantic add" .-> CACHE
  SUM -. "summary cache (TTL)" .-> CACHE
```

### Sequence (QA happy path)

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant API as API Gateway
  participant Ret as Retrieval
  participant LLM as LLM Generate
  participant Eval as Evaluation
  participant Cache as Redis

  User->>API: POST /ask_question
  API->>Cache: GET exact/semantic (index-versioned)
  alt cache hit
    Cache-->>API: QAResponse (cached)
    API-->>User: 200 OK (cached)
  else cache miss
    API->>Ret: hybrid_search(question)
    Ret-->>API: hits + diagnostics
    API->>API: optional rerank + threshold check
    opt refine or translate
      API->>Ret: hybrid_search(variant)
      Ret-->>API: extra hits
    end
    API->>LLM: generate(question + context)
    LLM-->>API: answer + citations
    API->>Eval: evaluate(heuristics/judge)
    Eval-->>API: metrics
    API->>Cache: SET exact + semantic + fingerprint (versioned)
    API-->>User: 200 OK (answer + confidence)
  end
```

## Key design points

- Hybrid retrieval with optional RRF union and reranker:
  - [`services.retrieval.app.hybrid`](services/retrieval/app/hybrid.py), [`services.retrieval.app.rerank`](services/retrieval/app/rerank.py)
- FAISS persistence + doc map; status/ops:
  - [`services.retrieval.app.faiss_store`](services/retrieval/app/faiss_store.py), [`/status`](services/retrieval/app/main.py), [`/debug/storage`](services/retrieval/app/main.py), [`/chunks_by_doc`](services/retrieval/app/main.py)
- Caching: exact + semantic + fingerprint; index-version invalidation on upload:
  - [`shared.cache`](shared/cache.py), used in [`services.api_gateway.app.routers.qa`](services/api_gateway/app/routers/qa.py) and [`services.api_gateway.app.routers.summary`](services/api_gateway/app/routers/summary.py); version bump in [`services.api_gateway.app.routers.upload`](services/api_gateway/app/routers/upload.py)
- LLM generation: router prompt, strict JSON parsing, citation heuristics:
  - [`services.llm_generate.app.main`](services/llm_generate/app/main.py), prompts in [`services.llm_generate.app.prompts`](services/llm_generate/app/prompts.py)
- Evaluation and confidence:
  - [`services.evaluation.app.metrics`](services/evaluation/app/metrics.py), [`services.evaluation.app.main`](services/evaluation/app/main.py), [`services.evaluation.app.confidence`](services/evaluation/app/confidence.py)
- Tracing/observability (opt-in):
  - [`shared.tracing`](shared/tracing.py) wired in services and routers
- Render/cloud specifics:
  - Retrieval requires persistent disk at /app/data for FAISS/doc_map; ensure EMBEDDING_DIM matches across services.
  - Gateway summary path requires both RETRIEVAL_URL and LLM_GENERATE_URL; otherwise local mode is used.