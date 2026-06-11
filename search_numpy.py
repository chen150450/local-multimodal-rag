#!/usr/bin/env python3
"""
RAG Semantic Search — Jina v5 + NumPy brute force cosine similarity

Replaces FAISS with pure numpy brute force.
Loads int8-quantized vectors from SQLite, dequantizes to float32,
does exact cosine similarity against all vectors in one BLAS call.

Scaling:
  - 1K  vectors: <1ms search, ~50MB float32
  - 1M  vectors: ~10ms search, ~1GB float32
  - 10M vectors: ~100ms search, ~20GB float32 (needs 32GB+ RAM)

Env:
  TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa  (avoids FlashAttention2 crash in Jina v5)

Configuration loaded from config.yaml via config_loader.
"""

import os, sys, json, time, sqlite3, argparse
import numpy as np

os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "sdpa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Import config loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_loader import (
        get_db_path, get_embedding_dim, get_jina_model_path,
        get_vllm_api_url, get_vllm_model_name, get_config
    )
    config = get_config()
    DB_PATH = get_db_path()
    MODEL_PATH = get_jina_model_path()
    EMBEDDING_DIM = get_embedding_dim()
    INT8_SCALE = config['embedding']['int8_scale']
except ImportError:
    # Fallback for standalone usage
    DB_PATH = os.environ.get("DB_PATH", "./file_index.db")
    MODEL_PATH = os.path.expanduser("~/models/jina-embeddings-v5-text-small/")
    EMBEDDING_DIM = 512
    INT8_SCALE = 127.0

EXT_GROUPS = {
    'text':    {'.txt','.md','.rst','.log','.csv','.tsv','.lrc'},
    'code':    {'.py','.js','.ts','.java','.c','.cpp','.h','.hpp','.cs','.go','.rs',
                '.rb','.php','.lua','.sh','.sql','.r','.m','.scala','.kt','.swift',
                '.dart','.ps1','.bat','.css','.scss','.html','.vue','.jsx','.tsx'},
    'config':  {'.json','.yaml','.yml','.toml','.xml','.ini','.cfg','.conf',
                '.env','.properties','.gitignore','.dockerignore','.editorconfig'},
    'pdf':     {'.pdf'},
    'image':   {'.jpg','.jpeg','.png','.gif','.bmp','.webp','.tiff','.tif','.svg','.ico'},
    'audio':   {'.mp3','.wav','.flac','.ogg','.m4a','.aac','.wma','.opus',
                '.wem','.amr','.aiff','.aif','.sfk','.mid','.midi','.rmi'},
    'ebook':   {'.epub','.mobi','.azw3','.fb2'},
    'djvu':    {'.djvu'},
    'office':  {'.doc','.docx','.xls','.xlsx','.ppt','.pptx','.odt','.ods','.odp','.rtf'},
    'archive': {'.zip','.rar','.7z','.tar','.gz','.bz2','.xz'},
}


def load_vectors():
    """Load int8 vectors from DB, dequantize to float32, normalize.

    Returns (vecs_f32, meta) where:
      vecs_f32: np.ndarray shape (N, 512) float32, L2-normalized
      meta:     dict with chunk_ids, paths, exts, chunk_indices, meta_json lists
    """
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    n = c.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    if n == 0:
        print("No vectors in database.")
        conn.close()
        return None, None

    c.execute("""
        SELECT r.id, f.path, f.ext, r.chunk_index, r.meta_json, r.vector
        FROM rag_chunks r JOIN files f ON r.file_id = f.id
        ORDER BY r.id
    """)

    vecs_int8 = np.empty((n, EMBEDDING_DIM), dtype=np.int8)
    meta = {
        "chunk_ids": [],
        "paths": [],
        "exts": [],
        "chunk_indices": [],
        "meta_json": [],
    }

    for i, row in enumerate(c):
        chunk_id, path, ext, chunk_idx, mj, vec_blob = row
        vecs_int8[i] = np.frombuffer(vec_blob, dtype=np.int8)
        meta["chunk_ids"].append(chunk_id)
        meta["paths"].append(path)
        meta["exts"].append(ext)
        meta["chunk_indices"].append(chunk_idx)
        meta["meta_json"].append(mj)
    conn.close()

    # Dequantize: float_val = int8_val / INT8_SCALE
    vecs_f32 = vecs_int8.astype(np.float32) / INT8_SCALE
    del vecs_int8

    # Re-normalize (original was unit-norm, but quantization adds ~0.4% L2 error)
    norms = np.linalg.norm(vecs_f32, axis=1, keepdims=True)
    np.maximum(norms, 1e-10, out=norms)
    vecs_f32 /= norms

    dt = time.time() - t0
    mem_mb = vecs_f32.nbytes / 1024**2
    print(f"Loaded {n:,} vectors ({mem_mb:.0f}MB float32) in {dt:.2f}s")
    return vecs_f32, meta


def search(vecs_f32, meta, query_vec, limit=20, ext_filter=None):
    """Brute force cosine similarity. query_vec must be L2-normalized float32 (512,).

    Returns list of dicts: {score, path, ext, location, chunk}
    """
    t0 = time.time()
    scores = vecs_f32 @ query_vec  # single BLAS call, ~10-100ms for millions
    order = np.argsort(-scores)    # descending

    results = []
    for idx in order:
        if ext_filter and meta["exts"][idx] not in ext_filter:
            continue

        mj = meta["meta_json"][idx]
        location = f"chunk {meta['chunk_indices'][idx]}"
        if mj:
            try:
                m = json.loads(mj)
                if m.get("pages"):
                    location = m["pages"]
                elif m.get("source") == "pdf_ocr" and m.get("ocr_pages", 1) > 1:
                    location = f"chunk {meta['chunk_indices'][idx]} / {m['ocr_pages']}页"
            except (json.JSONDecodeError, IndexError):
                pass

        results.append({
            "score": int(float(scores[idx]) * 1000),
            "path": meta["paths"][idx],
            "ext": meta["exts"][idx],
            "location": location,
            "chunk": meta["chunk_indices"][idx],
        })

        if len(results) >= limit:
            break

    dt = time.time() - t0
    print(f"Searched {len(scores):,} vectors in {dt*1000:.1f}ms")
    return results


def main():
    p = argparse.ArgumentParser(
        description="RAG Search — Jina v5 + NumPy brute force cosine similarity"
    )
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", type=int, default=20, help="Max results (default 20)")
    p.add_argument("--types", type=str, default="",
                   help="Filter: text,code,config,pdf,image,audio,ebook,office,archive")
    p.add_argument("--benchmark", action="store_true", help="Show detailed timing")
    p.add_argument("--config", type=str, default="", help="Path to config.yaml")
    args = p.parse_args()

    # Load custom config if specified
    if args.config:
        from config_loader import reset_config_cache, load_config
        reset_config_cache()
        load_config(args.config)

    # Build extension filter
    type_names = [t.strip() for t in args.types.split(",") if t.strip()]
    ext_filter = None
    if type_names:
        ext_filter = set()
        for t in type_names:
            if t in EXT_GROUPS:
                ext_filter.update(EXT_GROUPS[t])

    # ── Load ──────────────────────────────────────────────────
    vecs_f32, meta = load_vectors()
    if vecs_f32 is None:
        return

    # ── Embed query ───────────────────────────────────────────
    from jina_v5_embedding import JinaV5Embedder

    t_emb = time.time()
    embedder = JinaV5Embedder(
        model_path=MODEL_PATH, device="cuda", truncate_dim=EMBEDDING_DIM
    )
    q_vec = embedder.encode([args.query], task="retrieval", prompt_name="query")[0]
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm
    q_vec = q_vec.astype(np.float32)
    if args.benchmark:
        print(f"Embedded query in {time.time()-t_emb:.2f}s")

    # ── Search ────────────────────────────────────────────────
    results = search(vecs_f32, meta, q_vec, args.limit, ext_filter)

    if not results:
        print("No results found.")
        return

    print(f'\n🔍 Results for: "{args.query}" ({len(results)} chunks)\n')
    print(f'{"#":<3} {"Score":<8} {"Ext":<8} {"Location":<20} File')
    print("-" * 120)
    for i, r in enumerate(results, 1):
        path = r["path"] if len(r["path"]) <= 80 else "..." + r["path"][-77:]
        print(f'{i:<3} {r["score"]:<8} {r["ext"]:<8} {r["location"]:<20} {path}')
    print()


if __name__ == "__main__":
    main()