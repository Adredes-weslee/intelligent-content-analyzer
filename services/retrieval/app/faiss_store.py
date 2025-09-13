"""FAISS vector store management.

This module encapsulates a persistent FAISS index for dense retrieval.
It supports:
- Initializing/loading an index from disk, with metric configurable via settings.
- Adding vectors with explicit integer IDs mapped from chunk IDs.
- Removing vectors by chunk ID (using FAISS remove_ids).
- Searching by query vector and returning (chunk_id, score) pairs.
- Persisting and loading the chunk<->faiss_id mapping alongside the index.

The mapping and checksum/doc tracking share a JSON file defined by
`Settings.doc_map_path` with the structure:
{
  "checksums": { sha256: doc_id, ... },
  "files": { filename: { doc_id, checksum, chunks: [chunk_ids...] } },
  "chunk_to_faiss_id": { chunk_id: int, ... },
  "faiss_id_to_chunk": { int: chunk_id, ... },
  "faiss_dim": int,
  "faiss_metric": "ip"|"l2"
}
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from shared.settings import Settings

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore


_s = Settings()
_INDEX_PATH = _s.faiss_index_path
_DOC_MAP_PATH = _s.doc_map_path
_METRIC = (_s.faiss_metric or "ip").lower()
_NORMALIZE = bool(_s.faiss_normalize)

_index = None
_dim: int = _s.embedding_dim
_chunk_to_fid: Dict[str, int] = {}
_fid_to_chunk: Dict[int, str] = {}


def _ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_doc_map() -> dict:
    if os.path.exists(_DOC_MAP_PATH):
        try:
            with open(_DOC_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_doc_map(data: dict) -> None:
    _ensure_dirs(_DOC_MAP_PATH)
    try:
        with open(_DOC_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def init_index(dim: Optional[int] = None) -> None:
    """Load or create a FAISS index using configured metric.

    If an on-disk index exists, loads it and mapping; otherwise creates a new one.
    """
    global _index, _dim, _chunk_to_fid, _fid_to_chunk
    if faiss is None:
        return
    _dim = int(dim or _s.embedding_dim or 128)
    if os.path.exists(_INDEX_PATH):
        try:
            _index = faiss.read_index(_INDEX_PATH)
        except Exception:
            _index = None
    if _index is None:
        if _METRIC == "l2":
            base = faiss.IndexFlatL2(_dim)
        else:
            base = faiss.IndexFlatIP(_dim)
        _index = faiss.IndexIDMap2(base)
    # Load mapping
    m = _load_doc_map()
    _chunk_to_fid = {k: int(v) for k, v in m.get("chunk_to_faiss_id", {}).items()}
    _fid_to_chunk = {int(k): v for k, v in m.get("faiss_id_to_chunk", {}).items()}


def _persist() -> None:
    if faiss is None or _index is None:
        return
    _ensure_dirs(_INDEX_PATH)
    try:
        faiss.write_index(_index, _INDEX_PATH)
    except Exception:
        pass
    # Save mapping
    m = _load_doc_map()
    m["chunk_to_faiss_id"] = {k: int(v) for k, v in _chunk_to_fid.items()}
    m["faiss_id_to_chunk"] = {int(k): v for k, v in _fid_to_chunk.items()}
    m["faiss_dim"] = _dim
    m["faiss_metric"] = _METRIC
    _save_doc_map(m)


def _assign_ids(n: int) -> List[int]:
    """Assign n new integer IDs not in use (simple incremental allocator)."""
    used = set(_fid_to_chunk.keys())
    cur = 0 if not used else max(used) + 1
    ids: List[int] = []
    for _ in range(n):
        while cur in used:
            cur += 1
        ids.append(cur)
        cur += 1
    return ids


def add_vectors(chunk_ids: List[str], vectors: List[List[float]]) -> None:
    """Add vectors for chunk IDs, skipping those already present.

    Vectors are normalized if configured and the metric is IP.
    """
    if faiss is None or _index is None:
        return
    dim = len(vectors[0]) if vectors else _dim
    if dim != _dim:
        # Best-effort: attempt to accept different dimension by reinit
        init_index(dim)
    new_vecs: List[List[float]] = []
    for cid, vec in zip(chunk_ids, vectors):
        if cid in _chunk_to_fid:
            # skip existing; use upsert_vectors for replace
            continue
        v = np.array(vec, dtype="float32")
        if _NORMALIZE and _METRIC == "ip":
            n = np.linalg.norm(v) or 1.0
            v = v / n
        new_vecs.append(v.tolist())
    if not new_vecs:
        return
    faiss_ids = _assign_ids(len(new_vecs))
    # Rebuild mapping for order correspondence
    idx = 0
    for cid, vec in zip(chunk_ids, vectors):
        if cid in _chunk_to_fid:
            continue
        _chunk_to_fid[cid] = faiss_ids[idx]
        _fid_to_chunk[faiss_ids[idx]] = cid
        idx += 1
    xb = np.asarray(new_vecs, dtype="float32")
    ids = np.asarray(faiss_ids, dtype="int64")
    _index.add_with_ids(xb, ids)
    _persist()


def upsert_vectors(chunk_ids: List[str], vectors: List[List[float]]) -> None:
    """Replace vectors for the given chunk IDs if present, else add them."""
    if faiss is None or _index is None:
        return
    to_add_cids: List[str] = []
    to_add_vecs: List[List[float]] = []
    to_replace: List[int] = []
    new_for_replace: List[List[float]] = []
    for cid, vec in zip(chunk_ids, vectors):
        v = np.array(vec, dtype="float32")
        if _NORMALIZE and _METRIC == "ip":
            n = np.linalg.norm(v) or 1.0
            v = v / n
        if cid in _chunk_to_fid:
            to_replace.append(_chunk_to_fid[cid])
            new_for_replace.append(v.tolist())
        else:
            to_add_cids.append(cid)
            to_add_vecs.append(v.tolist())
    # Replace by removing then adding with same IDs
    if to_replace:
        ids = np.asarray(to_replace, dtype="int64")
        _index.remove_ids(ids)
        xb = np.asarray(new_for_replace, dtype="float32")
        _index.add_with_ids(xb, ids)
    if to_add_cids:
        faiss_ids = _assign_ids(len(to_add_cids))
        for cid, fid in zip(to_add_cids, faiss_ids):
            _chunk_to_fid[cid] = fid
            _fid_to_chunk[fid] = cid
        xb = np.asarray(to_add_vecs, dtype="float32")
        ids = np.asarray(faiss_ids, dtype="int64")
        _index.add_with_ids(xb, ids)
    _persist()


def remove_chunks(chunk_ids: Iterable[str]) -> None:
    if faiss is None or _index is None:
        return
    fids = [fid for cid in chunk_ids if (fid := _chunk_to_fid.get(cid)) is not None]
    if not fids:
        return
    _index.remove_ids(np.asarray(fids, dtype="int64"))
    for fid in fids:
        cid = _fid_to_chunk.pop(fid, None)
        if cid:
            _chunk_to_fid.pop(cid, None)
    _persist()


def search(vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
    """Search the FAISS index and return [(chunk_id, score), ...]."""
    if faiss is None or _index is None:
        return []
    v = np.asarray(vector, dtype="float32")
    if _NORMALIZE and _METRIC == "ip":
        n = np.linalg.norm(v) or 1.0
        v = v / n
    D, idxs = _index.search(v.reshape(1, -1), top_k)
    results: List[Tuple[str, float]] = []
    for dist, fid in zip(D[0], idxs[0]):
        if fid == -1:
            continue
        cid = _fid_to_chunk.get(int(fid))
        if cid is not None:
            results.append((cid, float(dist)))
    return results


def track_doc(filename: str, doc_id: str, checksum: str, chunk_ids: List[str]) -> None:
    """Update the doc map with file->doc mapping, checksum map, and chunk list."""
    m = _load_doc_map()
    m.setdefault("files", {})[filename] = {
        "doc_id": doc_id,
        "checksum": checksum,
        "chunks": chunk_ids,
    }
    m.setdefault("checksums", {})[checksum] = doc_id
    _save_doc_map(m)


def lookup_by_checksum(checksum: str) -> Optional[str]:
    m = _load_doc_map()
    return m.get("checksums", {}).get(checksum)


def lookup_file(filename: str) -> Optional[dict]:
    m = _load_doc_map()
    return m.get("files", {}).get(filename)


def update_file_entry(filename: str, checksum: str, chunk_ids: List[str]) -> None:
    m = _load_doc_map()
    if filename in m.get("files", {}):
        m["files"][filename]["checksum"] = checksum
        m["files"][filename]["chunks"] = chunk_ids
    _save_doc_map(m)
