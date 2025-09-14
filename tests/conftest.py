import os

# Deterministic, offline-friendly tests
os.environ.setdefault("OFFLINE_MODE", "1")
os.environ.setdefault("EVAL_LLM_ENABLED", "0")

# Retrieval/embeddings consistency and local FAISS paths
os.environ.setdefault("EMBEDDING_DIM", "768")
os.environ.setdefault("FAISS_INDEX_PATH", "data/faiss.index")
os.environ.setdefault("DOC_MAP_PATH", "data/doc_map.json")
