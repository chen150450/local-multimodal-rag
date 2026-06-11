#!/usr/bin/env python3
"""
RAG Pipeline — Jina v5 pure text + PaddleOCR CPU

Architecture (single process --resume starts everything):
- Jina v5 text-small (512-dim Matryoshka) for embedding
- PaddleOCR CPU mode, lazy-loaded per worker process
- Describer threads (OpenAI compatible API) for OCR-failed files
- OCR quality check: garbled/too-short results are skipped
- Video files are skipped entirely

Usage:
  python run_pipeline.py scan                      # 扫描文件
  python run_pipeline.py extract-meta              # 元数据提取
  python run_pipeline.py process                   # RAG 处理
  python run_pipeline.py search "关键词"           # 搜索
  python run_pipeline.py serve                     # 启动 Web 服务
  
  # 旧版参数兼容（直接运行 = process）
  python3 run_pipeline.py --force                  # Full reprocess, overwrite all
  python3 run_pipeline.py                          # Resume: skip done, continue
  python3 run_pipeline.py --limit 100              # Test with 100 files
  python3 run_pipeline.py --types text,code        # Only process certain types
  
  # Run independently of OpenClaw (gateway restart won't kill):
  nohup python3 run_pipeline.py process --force > pipeline.log 2>&1 &
  tail -f pipeline.log  # watch progress
"""
import os

# ============================================================
# 内存泄漏修复：强制 glibc 归还释放的内存给操作系统
# 必须在所有 import 之前设置
# ============================================================
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "0"
os.environ["MALLOC_MMAP_MAX_"] = "0"
os.environ["PYTHONMALLOC"] = "malloc"  # 绕过 pymalloc，让 glibc 直接管理

import sys
import time
import json
import sqlite3
import signal
import logging
import argparse
import shutil
import glob
import threading
import queue
import multiprocessing
import subprocess
import gc
import platform
import numpy as np

# Use fork — workers need to inherit parent context
# PaddleOCR uses separate subprocesses, so no CUDA fork issue
# 跨平台：Windows 不支持 fork，用 spawn
if platform.system() == 'Windows':
    multiprocessing.set_start_method('spawn', force=True)
else:
    multiprocessing.set_start_method('fork', force=True)
# 多进程队列用于跨进程传递 chunks
MP_QUEUE = None

# === 诊断：segfault 时自动打印 traceback ===
import faulthandler
import atexit
import resource

faulthandler.enable(all_threads=True)
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import (
    get_config, get_db_path, get_log_dir, get_embedding_dim,
    get_ocr_api_url, get_jina_model_path, get_vllm_api_url,
    get_vllm_model_name, get_env, get_pipeline_config
)
from extractors import extract, classify_file, ChunkData
from jina_v5_embedding import JinaV5Embedder
from describer import start_describers, stop_describers

# ============================================================
# Config (loaded from config.yaml)
# ============================================================
_config = get_config()
DB_PATH = get_db_path()
MODEL_PATH = get_jina_model_path()
EMBEDDING_DIM = get_embedding_dim()
LOG_DIR = get_log_dir()
OCR_API_URL = get_ocr_api_url()

# Pipeline config
_pipeline_config = get_pipeline_config()
BATCH_SIZE = _config['embedding']['batch_size']
CHUNK_QUEUE_SIZE = _pipeline_config['chunk_queue_size']
NUM_EXTRACTOR_THREADS = _pipeline_config['extractor_threads']
NUM_AUDIO_THREADS = _pipeline_config['audio_threads']
STREAM_BATCH_SIZE = _pipeline_config['stream_batch_size']

CHUNK_QUEUE = None  # Global queue, inherited by fork workers

# ============================================================
# Logging
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"rag_jina_v5_pipeline_{ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# === 诊断增强（在 LOG_DIR 定义之后） ===
_fault_log = open(os.path.join(LOG_DIR, "segfault_trace.log"), "a")
faulthandler.enable(file=_fault_log, all_threads=True)

def _close_fault_log():
    """关闭诊断日志文件句柄"""
    try:
        _fault_log.close()
    except Exception:
        pass

atexit.register(_close_fault_log)
atexit.register(lambda: log.info(
    f"🧠 Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB"
))


# ============================================================
# Database
# ============================================================
def init_db():
    """Create tables if not exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # rag_chunks: 存储文件 chunks 的向量和元数据
    c.execute("""CREATE TABLE IF NOT EXISTS rag_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        chunk_index INTEGER,
        meta_json TEXT,         -- JSON: source, ocr_pages, language, etc.
        vector BLOB,            -- int8 quantized embedding (512 bytes, 512-dim Matryoshka)
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON rag_chunks(file_id)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_file_idx ON rag_chunks(file_id, chunk_index)")
    
    # rag_status: 断点续传状态表（支持优雅停止）
    c.execute("""CREATE TABLE IF NOT EXISTS rag_status (
        file_id INTEGER PRIMARY KEY,
        status TEXT DEFAULT 'pending',  -- pending/processing/done/error
        chunks_written INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_rag_status_status ON rag_status(status)")
    
    # rag_descriptions: 视觉模型生成的描述（OCR失败后的fallback）
    c.execute("""CREATE TABLE IF NOT EXISTS rag_descriptions (
        file_id INTEGER PRIMARY KEY,
        description TEXT,
        source TEXT DEFAULT 'unknown',  -- qwen_api, lmstudio, qwen_failed, lmstudio_failed
        embedded INTEGER DEFAULT 0,     -- 是否已嵌入到 rag_chunks
        retry_count INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id)
    )""")
    
    conn.commit()
    conn.close()
    log.info("DB tables ready (rag_chunks + rag_status + rag_descriptions)")

    # Clean orphaned temp dirs
    import tempfile
    tmp_base = tempfile.gettempdir()
    orphans = glob.glob(os.path.join(tmp_base, 'rag_pdf_*'))
    if orphans:
        for d in orphans:
            shutil.rmtree(d, ignore_errors=True)
        log.info(f"Cleaned {len(orphans)} orphaned temp dirs")


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# ============================================================
# File type → ext set mapping
# ============================================================
from extractors import (TEXT_EXTS, CODE_EXTS, CONFIG_EXTS, PDF_EXTS,
                        IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS, EBOOK_EXTS, DJVU_EXTS,
                        OFFICE_EXTS, ARCHIVE_EXTS)

TYPE_TO_EXTS = {
    'text': TEXT_EXTS,
    'code': CODE_EXTS,
    'config': CONFIG_EXTS,
    'pdf_text': PDF_EXTS,
    'pdf_image': PDF_EXTS,
    'image': IMAGE_EXTS,
    'video': VIDEO_EXTS,   # skipped in extractor, but still queryable
    'audio': AUDIO_EXTS,
    'ebook': EBOOK_EXTS,
    'djvu': DJVU_EXTS,
    'office': OFFICE_EXTS,
    'archive': ARCHIVE_EXTS,
}


# ============================================================
# CPU Producer: File Extraction (multiprocessing.Pool)
# ============================================================
def _process_batch_wrapper(args_tuple):
    """Wrapper for imap_unordered - unpacks single tuple argument."""
    batch, producer_id, skip_ids, audio_threads = args_tuple
    return _process_file_batch(batch, producer_id, skip_ids, audio_threads)

def _process_file_batch(batch: List[tuple], producer_id: int, skip_ids: set, audio_threads: int = 4) -> tuple:
    """Process a batch of files in a separate process.
    
    Audio files use ThreadPoolExecutor for parallel CPU transcription.
    Other files are processed serially (GPU-bound, serialized by lock anyway).
    """
    import json
    stats = {'extracted': 0, 'chunks': 0, 'errors': 0, 'skipped': 0, 'chars': 0}
    chunks_data = []
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extractors import extract, classify_file, AUDIO_EXTS
    
    # Separate audio files for parallel processing
    audio_items = []
    other_items = []
    for row in batch:
        file_id, path, name, ext, size, media_meta, mtime = row
        if file_id in skip_ids:
            stats['skipped'] += 1
            continue
        if ext and ext.lower() in AUDIO_EXTS:
            audio_items.append(row)
        else:
            other_items.append(row)
    
    # 动态超时：需要 OCR 的文件给更多时间（OCR API 内部 timeout=600s）
    def _get_file_timeout(ext, file_type):
        if ext and ext.lower() in ('.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'):
            return 300  # 5 分钟（PDF 渲染 + OCR）
        return 30  # 普通文件 30 秒

    def _extract_one(row):
        """Extract chunks from a single file. Returns (stats_dict, chunks_list)."""
        file_id, path, name, ext, size, media_meta, mtime = row
        local_stats = {'extracted': 0, 'chunks': 0, 'errors': 0, 'skipped': 0, 'chars': 0}
        local_chunks = []
        file_type = classify_file(ext, media_meta)
        try:
            # 使用线程+超时防止单文件卡死
            import threading
            result_holder = [None, None]  # [chunks, exception]
            
            def _do_extract():
                try:
                    result_holder[0] = extract(file_id, path, ext, file_type, media_meta, size or 0)
                except Exception as e:
                    result_holder[1] = e
            
            file_timeout = _get_file_timeout(ext, file_type)
            t = threading.Thread(target=_do_extract, daemon=True)
            t.start()
            t.join(timeout=file_timeout)
            
            if t.is_alive():
                log.warning(f"File extraction timed out ({file_timeout}s): {path}")
                local_stats['errors'] += 1
                return local_stats, local_chunks
            
            if result_holder[1] is not None:
                raise result_holder[1]
            
            chunks = result_holder[0]
            if not chunks:
                local_stats['skipped'] += 1
            else:
                for chunk in chunks:
                    chunk.file_size = size or 0
                    chunk.file_mtime = mtime or 0.0
                    local_chunks.append({
                        'file_id': chunk.file_id,
                        'chunk_index': chunk.chunk_index,
                        'text': chunk.text,
                        'char_count': chunk.char_count,
                        'meta': chunk.meta,
                        'file_size': chunk.file_size,
                        'file_mtime': chunk.file_mtime
                    })
                    local_stats['chars'] += chunk.char_count
                local_stats['extracted'] += 1
                local_stats['chunks'] += len(chunks)
        except Exception:
            local_stats['errors'] += 1
        return local_stats, local_chunks
    
    # Process non-audio files serially (with per-file timeout)
    for row in other_items:
        s, c = _extract_one(row)
        for k in stats: stats[k] += s.get(k, 0)
        chunks_data.extend(c)
    
    # 回收 Python 碎片内存，防止 OOM
    import gc
    gc.collect()
    # 强制 glibc 归还内存给 OS（PaddleOCR C++ 层泄漏的补救）
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    
    # Process audio files in parallel (CPU-bound, no GPU contention)
    if audio_items:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=audio_threads) as executor:
            futures = {executor.submit(_extract_one, row): row for row in audio_items}
            for future in as_completed(futures):
                s, c = future.result()
                for k in stats: stats[k] += s.get(k, 0)
                chunks_data.extend(c)
    
    return (producer_id, stats, chunks_data)


# ============================================================
# GPU Consumer: Jina v5 batch embedding
# ============================================================
class GPUConsumer:
    """GPU 嵌入消费者。
    
    当前流式模式下只使用 load_model() 和 _embed_text_batch()。
    其他方法（run, _process_batch, _flush_batch, _write_batch, _log_progress）
    是旧版队列模式遗留，保留供参考但不再调用。
    """
    """Main thread: loads Jina v5, embeds text, writes DB."""

    def __init__(self, mp_queue: multiprocessing.Queue, num_producers: int = 1):
        self.mp_queue = mp_queue
        self.num_producers = num_producers
        self.embedder = None
        self.stats = {'embedded': 0, 'batches': 0, 'write_errors': 0, 'total_chars': 0, 'files_done': 0}
        self._stop = False
        self._db_conn = None
        self._start_time = None
        self._total_files = 0
        self._est_chunks_per_file = 0
        self._file_meta = {}
        self._file_chunk_count = {}  # file_id → chunks written
        self._completed_files = 0  # files fully written
        self._producer_stats = {}  # producer_id → stats (从 STATS 消息汇总)

    def _get_db(self):
        if self._db_conn is None:
            self._db_conn = get_db()
        return self._db_conn

    def load_model(self):
        """Connect to vLLM embedding service."""
        log.info(f"Connecting to vLLM embedding service...")
        t0 = time.time()
        self.embedder = JinaV5Embedder(
            batch_size=BATCH_SIZE,
            truncate_dim=EMBEDDING_DIM,
        )
        # Check vLLM service availability
        self.embedder._check_service()
        load_time = time.time() - t0
        log.info(f"✅ Connected to vLLM embedding service in {load_time:.1f}s")

    def _embed_text_batch(self, texts: List[str], encode_batch_size: int = None) -> np.ndarray:
        """Embed text chunks via vLLM API.
        
        Args:
            texts: List of texts to encode
            encode_batch_size: Internal batch size (default: BATCH_SIZE)
        """
        # 使用全局 BATCH_SIZE 作为默认值
        if encode_batch_size is None:
            encode_batch_size = BATCH_SIZE
        
        log.info(f"[Jina] Embedding {len(texts)} texts via vLLM API (batch_size={encode_batch_size})")
        
        result = self.embedder.encode(texts, task='retrieval', prompt_name='document', batch_size=encode_batch_size)
        
        log.info(f"[Jina] Got {result.shape[0]} embeddings, dim={result.shape[1]}")
        
        return result

    def _quantize_to_int8(self, vectors: np.ndarray) -> bytes:
        """Global scalar quantization: L2-normed vectors * 127 -> int8."""
        truncated = vectors[:, :EMBEDDING_DIM]
        quantized = np.clip(np.round(truncated * 127), -127, 127).astype(np.int8)
        return quantized.tobytes()

    def _write_batch(self, chunks: List[ChunkData], embeddings: np.ndarray):
        """Write batch to DB."""
        conn = self._get_db()
        c = conn.cursor()

        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_file_idx ON rag_chunks(file_id, chunk_index)")
        except Exception:
            pass

        # Delete old chunks for these files first (prevents zombie chunks)
        file_ids = set(chunk.file_id for chunk in chunks)
        for fid in file_ids:
            c.execute("DELETE FROM rag_chunks WHERE file_id = ?", (fid,))

        for chunk, emb in zip(chunks, embeddings):
            vec_blob = self._quantize_to_int8(emb.reshape(1, -1))
            # Store chunk text in meta for cross-encoder re-ranking
            # (text_preview column was removed from DB, so we embed it in meta_json)
            meta_dict = dict(chunk.meta) if chunk.meta else {}
            if chunk.text:
                # 500 chars = ~165 中文字, 足够 cross-encoder 判断语义
                # 330 万 chunks × 500 chars ≈ +1.7GB DB
                meta_dict['_text'] = chunk.text[:500]
            try:
                c.execute("""INSERT INTO rag_chunks 
                            (file_id, chunk_index, meta_json, vector)
                            VALUES (?, ?, ?, ?)""",
                          (chunk.file_id, chunk.chunk_index,
                           json.dumps(meta_dict, ensure_ascii=False) if meta_dict else None,
                           vec_blob))
            except Exception as e:
                self.stats['write_errors'] += 1
                log.warning(f"DB write error: {e}")

        if self.stats['embedded'] > 0 and self.stats['embedded'] % 5000 == 0:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

        conn.commit()

    def run(self):
        """Main embedding loop — all chunks are text."""
        batch_chunks: List[ChunkData] = []
        last_log = time.time()
        producers_done = 0
        self._start_time = time.time()

        while not self._stop:
            try:
                item = self.mp_queue.get(timeout=5)
            except queue.Empty:
                if producers_done >= self.num_producers:
                    break
                continue

            # 处理 STATS 消息（生产者结束）
            if isinstance(item, tuple) and item[0] == 'STATS':
                producer_id, stats = item[1], item[2]
                self._producer_stats[producer_id] = stats
                producers_done += 1
                log.info(f"📦 Producer {producer_id} done ({producers_done}/{self.num_producers}): {stats}")
                if producers_done >= self.num_producers and batch_chunks:
                    self._process_batch(batch_chunks)
                    batch_chunks = []
                continue

            batch_chunks.append(item)
            if item.file_id not in self._file_meta:
                self._file_meta[item.file_id] = (item.file_size, item.file_mtime)

            if len(batch_chunks) >= BATCH_SIZE:
                self._flush_batch(batch_chunks)
                batch_chunks = []

            # Progress log every 30 seconds
            now = time.time()
            if now - last_log > 30:
                self._log_progress()
                last_log = now

        if batch_chunks:
            self._flush_batch(batch_chunks)

        # Final log
        self._log_progress(final=True)

        # Cleanup
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        if self._db_conn:
            try:
                self._db_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            self._db_conn.close()
            self._db_conn = None

        # Clean shared temp dir
        shared_tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'rag_pdf_shared')
        if os.path.isdir(shared_tmp):
            shutil.rmtree(shared_tmp, ignore_errors=True)
            log.info("🗑️ Cleaned shared temp dir")

        log.info(f"🔥 GPU consumer done: {self.stats}")

    def _flush_batch(self, batch_chunks: List[ChunkData]):
        """Process and write a batch."""
        try:
            self._process_batch(batch_chunks)
        except Exception as e:
            log.error(f'GPU batch crash: {e}')
            for c in batch_chunks:
                log.error(f'  Failed file_id={c.file_id} chunk={c.chunk_index}')
            self.stats['write_errors'] += len(batch_chunks)
            import torch
            torch.cuda.empty_cache()
            import gc; gc.collect()

    def _log_progress(self, final=False):
        """Log progress with files/chunks/chars throughput and ETA."""
        import torch
        elapsed = time.time() - self._start_time
        embedded = self.stats['embedded']
        total_chars = self.stats['total_chars']

        if elapsed <= 0:
            return

        # Count completed files from DB (distinct file_ids with chunks written)
        completed_files = 0
        try:
            conn = self._get_db()
            c = conn.cursor()
            c.execute("SELECT COUNT(DISTINCT file_id) FROM rag_chunks")
            completed_files = c.fetchone()[0]
        except Exception:
            completed_files = len(self._file_meta)  # fallback

        # Throughput rates
        files_per_s = completed_files / elapsed if completed_files > 0 else 0
        chunks_per_s = embedded / elapsed
        chars_per_s = total_chars / elapsed

        # ETA: based on files (more stable)
        total_files = self._total_files
        if total_files > 0 and completed_files > 5:
            remaining = max(0, total_files - completed_files)
            eta_s = remaining / files_per_s if files_per_s > 0 else 0
            pct = completed_files / total_files * 100
            eta_h = int(eta_s // 3600)
            eta_m = int((eta_s % 3600) // 60)
            eta_str = f"{eta_h}h{eta_m}m" if eta_h > 0 else f"{eta_m}min"
        else:
            eta_str = "..."
            pct = 0

        # VRAM
        free, total = torch.cuda.mem_get_info()
        vram_gb = round((total - free) / 1e9, 1)

        tag = "📊 FINAL" if final else "📊"
        log.info(
            f"{tag} {completed_files:,}/{total_files:,} files ({pct:.1f}%) | "
            f"{embedded:,} chunks | {total_chars:,} chars | "
            f"{files_per_s:.2f} files/s | {chunks_per_s:.1f} chunks/s | {chars_per_s:,.0f} chars/s | "
            f"ETA: {eta_str} | queue: {self.mp_queue.qsize()} | VRAM: {vram_gb} GB"
        )

    def _process_batch(self, chunks: List[ChunkData]):
        """Embed text chunks with Jina v5 and write to DB."""
        try:
            texts = [c.text for c in chunks]
            embs = self._embed_text_batch(texts)
            self._write_batch(chunks, embs)
            self.stats['embedded'] += len(chunks)
            self.stats['total_chars'] += sum(c.char_count for c in chunks)
            self.stats['batches'] += 1

        except Exception as e:
            log.error(f"Embedding batch error: {e}")
            file_ids = set(c.file_id for c in chunks)
            for fid in file_ids:
                log.error(f'  Failed file_id={fid}')
            self.stats['write_errors'] += len(chunks)

        # Cleanup
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    def stop(self):
        self._stop = True


# ============================================================
# Main
# ============================================================
# vLLM embedding service process
VLLM_PROC = None

def start_vllm_service():
    """Start vLLM embedding service as subprocess."""
    global VLLM_PROC
    import subprocess
    import requests
    
    vllm_config = _config['embedding']['vllm']
    vllm_dir = os.path.expanduser("~/vllm")
    model_path = get_jina_model_path()
    vllm_bin = os.path.join(vllm_dir, ".venv/bin/vllm")
    
    # Check if already running
    vllm_url = get_vllm_api_url()
    try:
        resp = requests.get(f"{vllm_url}/health", timeout=2)
        if resp.status_code == 200:
            log.info("✅ vLLM service already running")
            return True
    except:
        pass
    
    log.info("🚀 Starting vLLM embedding service...")
    
    cmd = [
        vllm_bin, "serve", model_path,
        "--port", str(vllm_config['port']),
        "--dtype", vllm_config['dtype'],
        "--trust-remote-code",
        "--gpu-memory-utilization", str(vllm_config['gpu_memory_utilization']),
        "--max-num-seqs", "32",
        "--max-model-len", str(vllm_config['max_model_len']),
        "--enforce-eager",  # 禁用 CUDA graphs，省显存
        "--hf-overrides", '{"architectures": ["TransformersEmbeddingModel"]}'
    ]
    
    VLLM_PROC = subprocess.Popen(
        cmd,
        cwd=vllm_dir,
        stdout=subprocess.DEVNULL,  # 不阻塞 vLLM（pipe 满会导致卡死）
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # 独立进程组，方便清理所有子进程
    )
    
    # Wait for service to be ready (max 120 seconds)
    for i in range(60):
        time.sleep(2)
        try:
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            if resp.status_code == 200:
                log.info(f"✅ vLLM service ready (took {(i+1)*2}s)")
                return True
        except:
            pass
        # Check if process died
        if VLLM_PROC.poll() is not None:
            log.error("❌ vLLM process died during startup")
            return False
    
    log.error("❌ vLLM service failed to start within 120s")
    return False


def stop_vllm_service():
    """Stop vLLM embedding service and all its child processes."""
    global VLLM_PROC
    if VLLM_PROC and VLLM_PROC.poll() is None:
        log.info("Stopping vLLM service...")
        # Kill entire process group (主进程 + EngineCore + resource_tracker)
        try:
            os.killpg(os.getpgid(VLLM_PROC.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            VLLM_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(VLLM_PROC.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            VLLM_PROC.wait()
        log.info("✅ vLLM service stopped")
    VLLM_PROC = None
    # 清理残留的孤儿 EngineCore 进程
    try:
        subprocess.run(['pkill', '-f', 'VLLM::EngineCore'], 
                       timeout=5, capture_output=True)
    except Exception:
        pass


# ============================================================
# OCR API service process (PaddleOCR GPU)
OCR_PROC = None

def start_ocr_service():
    """Start PaddleOCR GPU API server as a persistent subprocess."""
    global OCR_PROC
    
    # Check if already running
    try:
        import requests as _req
        resp = _req.get(f"{OCR_API_URL}/health", timeout=2, proxies={"http": None, "https": None})
        if resp.status_code == 200:
            log.info(f"✅ OCR API server already running at {OCR_API_URL}")
            return True
    except Exception:
        pass
    
    log.info("🚀 Starting OCR API server (PaddleOCR GPU)...")
    
    ocr_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr_server.py')
    ocr_config = _config['ocr']
    port = ocr_config['port']
    
    # Redirect stdout/stderr to log file to prevent pipe buffer blocking
    ocr_log_path = os.path.join(LOG_DIR, 'ocr_server.log')
    ocr_log_file = open(ocr_log_path, 'w')
    
    OCR_PROC = subprocess.Popen(
        [sys.executable, ocr_script, '--port', str(port)],
        stdout=ocr_log_file,
        stderr=ocr_log_file,
        preexec_fn=os.setsid,
    )
    
    # Wait for ready (up to 120s for model loading)
    import requests as _req
    deadline = time.time() + 120
    while time.time() < deadline:
        if OCR_PROC.poll() is not None:
            log.error(f"❌ OCR server died during startup")
            return False
        try:
            resp = _req.get(f"{OCR_API_URL}/health", timeout=2, proxies={"http": None, "https": None})
            if resp.status_code == 200:
                log.info(f"✅ OCR API server ready at {OCR_API_URL}")
                return True
        except Exception:
            pass
        time.sleep(2)
    
    log.error("❌ OCR server startup timed out")
    return False


def stop_ocr_service():
    """Stop OCR API server."""
    global OCR_PROC
    if OCR_PROC and OCR_PROC.poll() is None:
        log.info("Stopping OCR API server...")
        try:
            os.killpg(os.getpgid(OCR_PROC.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            OCR_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(OCR_PROC.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            OCR_PROC.wait()
        log.info("✅ OCR API server stopped")
    OCR_PROC = None


def run_rag_process():
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description="RAG Pipeline — Jina v5 + PaddleOCR CPU")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--types", type=str, default="", help="Comma-separated file types")
    parser.add_argument("--ids-file", type=str, default="", help="File with file IDs")
    parser.add_argument("--resume", action="store_true", help="Skip already processed")
    parser.add_argument("--force", action="store_true", help="Force reprocess all (delete old rag_chunks first)")
    parser.add_argument("--no-faiss", action="store_true", help="Skip FAISS index build at the end")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=NUM_EXTRACTOR_THREADS)
    parser.add_argument("--audio-threads", type=int, default=NUM_AUDIO_THREADS, help="Audio transcription threads per worker process")
    parser.add_argument("--config", type=str, default="", help="Path to config.yaml (default: same dir as script)")
    args = parser.parse_args()

    # Load custom config if specified
    if args.config:
        from config_loader import reset_config_cache, load_config
        reset_config_cache()
        load_config(args.config)

    BATCH_SIZE = args.batch_size
    types_filter = [t.strip() for t in args.types.split(',') if t.strip()] if args.types else None

    log.info("=" * 60)
    log.info("🚀 RAG Pipeline — Jina v5 + PaddleOCR CPU")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"Embedding dim: {EMBEDDING_DIM} | Batch: {BATCH_SIZE} | Threads: {args.threads}")
    log.info(f"Types: {types_filter or 'all'} | Limit: {args.limit or 'none'}")
    log.info(f"Force: {args.force} | No FAISS: {args.no_faiss}")
    log.info("=" * 60)

    # Start vLLM embedding service
    if not start_vllm_service():
        log.error("Failed to start vLLM service, exiting")
        sys.exit(1)

    # Start OCR API server (PaddleOCR GPU)
    if not start_ocr_service():
        log.error("Failed to start OCR API server, exiting")
        stop_vllm_service()
        sys.exit(1)

    init_db()

    # ---- Phase 0: Maintenance ----
    conn = get_db()
    c = conn.cursor()

    # 清理上次中断时留下的 processing 状态（部分 chunks）
    c.execute("SELECT file_id FROM rag_status WHERE status = 'processing'")
    processing_files = [row[0] for row in c.fetchall()]
    if processing_files:
        placeholders = ','.join(['?'] * len(processing_files))
        c.execute(f"DELETE FROM rag_chunks WHERE file_id IN ({placeholders})", processing_files)
        del_chunks = c.rowcount
        c.execute(f"DELETE FROM rag_status WHERE file_id IN ({placeholders})", processing_files)
        conn.commit()
        log.info(f"🧹 Cleaned {del_chunks} partial chunks from {len(processing_files)} interrupted files (status=processing)")

    if args.force:
        # Force mode: delete ALL rag_chunks + rag_status + rag_descriptions + rag_ocr_status
        c.execute("SELECT COUNT(DISTINCT file_id), COUNT(*) FROM rag_chunks")
        old_files, old_chunks = c.fetchone()
        c.execute("DELETE FROM rag_chunks")
        c.execute("DELETE FROM rag_status")
        c.execute("DELETE FROM rag_descriptions")
        c.execute("DELETE FROM rag_ocr_status")
        conn.commit()
        log.info(f"🗑️ Force mode: deleted {old_files:,} files, {old_chunks:,} chunks + rag_status + rag_descriptions + rag_ocr_status")
        skip_ids = set()
    else:
        # Incremental: clean deleted files, skip done files
        c.execute("DELETE FROM rag_chunks WHERE file_id IN (SELECT id FROM files WHERE is_deleted=1)")
        del_chunks = c.rowcount
        c.execute("DELETE FROM rag_status WHERE file_id IN (SELECT id FROM files WHERE is_deleted=1)")
        del_status = c.rowcount
        conn.commit()
        if del_chunks:
            log.info(f"🧹 Cleaned {del_chunks} chunks + {del_status} status from deleted files")

        # 使用 rag_status.status='done' 判断已处理文件
        c.execute("SELECT file_id FROM rag_status WHERE status = 'done'")
        skip_ids = set(row[0] for row in c.fetchall())
        if skip_ids:
            log.info(f"⏭️ Skipping {len(skip_ids):,} already-processed files (rag_status.status='done')")
        else:
            log.info("🆕 Fresh run — no previously processed files")

        # 增加 rag_ocr_status 检查（跳过 OCR 失败的文件）
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rag_ocr_status'")
        if c.fetchone():
            c.execute("SELECT file_id FROM rag_ocr_status")
            ocr_skip_ids = set(row[0] for row in c.fetchall())
            if ocr_skip_ids:
                skip_ids.update(ocr_skip_ids)
                log.info(f"⏭️ Skipping {len(ocr_skip_ids):,} OCR-insufficient files (rag_ocr_status)")

    # Build query
    if types_filter:
        expanded = []
        for t in types_filter:
            if t == 'pdf':
                expanded.extend(['pdf_text', 'pdf_image'])
            else:
                expanded.append(t)
        types_filter = expanded

        exts = set()
        for t in types_filter:
            if t in TYPE_TO_EXTS:
                exts.update(TYPE_TO_EXTS[t])
        if exts:
            placeholders = ','.join(['?'] * len(exts))
            if skip_ids:
                if len(skip_ids) > 10000:
                    c.execute("CREATE TEMP TABLE IF NOT EXISTS _skip_ids (file_id INTEGER PRIMARY KEY)")
                    c.executemany("INSERT OR IGNORE INTO _skip_ids VALUES (?)", [(fid,) for fid in skip_ids])
                    c.execute(f"""SELECT id, path, name, ext, size, media_meta, mtime
                                 FROM files WHERE is_deleted=0 
                                 AND LOWER(ext) IN ({placeholders})
                                 AND id NOT IN (SELECT file_id FROM _skip_ids)
                                 ORDER BY id""", list(exts))
                else:
                    skip_ph = ','.join(['?'] * len(skip_ids))
                    c.execute(f"""SELECT id, path, name, ext, size, media_meta, mtime
                                 FROM files WHERE is_deleted=0 
                                 AND LOWER(ext) IN ({placeholders})
                                 AND id NOT IN ({skip_ph})
                                 ORDER BY id""", list(exts) + list(skip_ids))
            else:
                c.execute(f"""SELECT id, path, name, ext, size, media_meta, mtime
                             FROM files WHERE is_deleted=0  AND LOWER(ext) IN ({placeholders})
                             ORDER BY id""", list(exts))
            rows = c.fetchall()
            file_rows = []
            for fid, fpath, fname, fext, fsize, fmeta, fmtime in rows:
                if fext.lower() == '.pdf':
                    ft = classify_file(fext, fmeta)
                    if ft in types_filter:
                        file_rows.append((fid, fpath, fname, fext, fsize, fmeta, fmtime))
                else:
                    file_rows.append((fid, fpath, fname, fext, fsize, fmeta, fmtime))
        else:
            file_rows = []
    else:
        if skip_ids:
            if len(skip_ids) > 10000:
                c.execute("CREATE TEMP TABLE IF NOT EXISTS _skip_ids (file_id INTEGER PRIMARY KEY)")
                c.executemany("INSERT OR IGNORE INTO _skip_ids VALUES (?)", [(fid,) for fid in skip_ids])
                c.execute("""SELECT id, path, name, ext, size, media_meta, mtime
                             FROM files WHERE is_deleted=0 
                             AND id NOT IN (SELECT file_id FROM _skip_ids) ORDER BY id""")
            else:
                placeholders = ','.join(['?'] * len(skip_ids))
                c.execute(f"""SELECT id, path, name, ext, size, media_meta, mtime
                             FROM files WHERE is_deleted=0 
                             AND id NOT IN ({placeholders}) ORDER BY id""", list(skip_ids))
        else:
            c.execute("""SELECT id, path, name, ext, size, media_meta, mtime
                         FROM files WHERE is_deleted=0  ORDER BY id""")
        file_rows = c.fetchall()

    conn.close()

    # Skip no-extension files
    no_ext = [r for r in file_rows if not r[3]]
    file_rows = [r for r in file_rows if r[3]]
    if no_ext:
        log.info(f"⏭️ Skipping {len(no_ext):,} files with no extension")

    # Only process files in monitored directories (from config)
    monitor_dirs = _config.get('monitor', {}).get('directories', [])
    if monitor_dirs:
        def _is_rag_path(p):
            return any(p.startswith(d) for d in monitor_dirs)
        file_rows = [r for r in file_rows if _is_rag_path(r[1])]
        log.info(f"After directory filter: {len(file_rows):,} files")

    log.info(f"Files to process: {len(file_rows):,}")

    if args.ids_file:
        with open(args.ids_file) as f:
            id_set = set(int(line.strip()) for line in f if line.strip())
        file_rows = [r for r in file_rows if r[0] in id_set]
        log.info(f"Filtered by ids-file: {len(file_rows):,} files")

    if not file_rows:
        log.info("Nothing to do!")
        return
    # ---- Phase 2: Streaming pipeline — process files in small batches, embed + write DB immediately ----
    # 改成小 batch 流式处理，每处理完一批就立刻 embed + 写 DB
    n_processes = min(args.threads, len(file_rows))
    file_rows = list(file_rows)  # 硝保是列表，支持切片
    file_batches = [file_rows[i:i+STREAM_BATCH_SIZE] for i in range(0, len(file_rows), STREAM_BATCH_SIZE)]
    log.info(f"Total {len(file_batches):,} batches (batch_size={STREAM_BATCH_SIZE})")

    log.info(f"Starting {n_processes} worker processes (audio_threads={args.audio_threads} per worker)...")
    
    # PaddleOCR is now lazy-loaded per worker process (CPU mode, no daemon needed)

    # Start describer threads (Qwen + LMStudio, process OCR-failed files)
    describer_threads = start_describers()
    
    pool = multiprocessing.Pool(processes=n_processes)
    
    # GPU consumer in main process - load model first
    gpu = GPUConsumer(mp_queue=None, num_producers=n_processes)
    gpu._total_files = len(file_rows)
    gpu.load_model()
    log.info("✅ Jina v5 loaded, ready for streaming embedding")
    
    def signal_handler(sig, frame):
        log.info("⚠️ Shutdown signal received, cleaning up...")
        pool.terminate()
        pool.join()
        stop_describers(describer_threads, timeout=30)
        stop_vllm_service()
        log.info("👋 Pipeline shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    t0 = time.time()
    
    # 流式处理：用 imap_unordered 每收到一个 batch 结果就立刻 embed + 写 DB
    total_stats = {'extracted': 0, 'chunks': 0, 'errors': 0, 'skipped': 0, 'chars': 0}
    written_files = 0
    written_chunks = 0
    conn = get_db()
    c = conn.cursor()
    
    log.info(f"Starting streaming processing of {len(file_batches):,} batches...")
    
    def embed_and_write_batch(chunks_data):
        """立刻 embed + 写 DB（包含错误处理）+ 顺带嵌入视觉描述"""
        if not chunks_data:
            # 即使文件 chunks 为空，也要消费积压的 rag_descriptions
            desc_embedded = 0
            try:
                c.execute("""
                    SELECT id, file_id, description 
                    FROM rag_descriptions 
                    WHERE embedded = 0 AND source NOT LIKE '%_failed'
                    ORDER BY created_at ASC
                    LIMIT 8
                """)
                desc_rows = c.fetchall()
                if desc_rows:
                    desc_texts = [row[2] for row in desc_rows]
                    desc_embeddings = gpu._embed_text_batch(desc_texts)
                    desc_quantized = np.clip(np.round(desc_embeddings[:, :EMBEDDING_DIM] * 127), -127, 127).astype(np.int8)
                    c.execute("BEGIN TRANSACTION")
                    for idx, (desc_id, file_id, description) in enumerate(desc_rows):
                        meta = json.dumps({
                            'source': 'vision_description',
                            'description': description[:200] + '...' if len(description) > 200 else description
                        }, ensure_ascii=False)
                        vec_blob = desc_quantized[idx].tobytes()
                        c.execute("""
                            INSERT OR REPLACE INTO rag_chunks 
                            (file_id, chunk_index, meta_json, vector)
                            VALUES (?, -1, ?, ?)
                        """, (file_id, meta, vec_blob))
                        c.execute("UPDATE rag_descriptions SET embedded = 1 WHERE id = ?", (desc_id,))
                        desc_embedded += 1
                    conn.commit()
                    if desc_embedded > 0:
                        log.info(f"[Vision] Embedded {desc_embedded} descriptions (no file chunks)")
            except Exception as e:
                log.error(f"Description embed error (no chunks): {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            return 0, desc_embedded
        try:
            texts = [chunk['text'] for chunk in chunks_data]
            
            # 从 rag_descriptions 取待嵌入的视觉描述（每次最多8条）
            desc_rows = []
            try:
                c.execute("""
                    SELECT id, file_id, description 
                    FROM rag_descriptions 
                    WHERE embedded = 0 AND source NOT LIKE '%_failed'
                    ORDER BY created_at ASC
                    LIMIT 8
                """)
                desc_rows = c.fetchall()
            except Exception:
                pass
            
            # 合并文本：文件 chunks + 视觉描述
            all_texts = texts + [row[2] for row in desc_rows]
            embeddings = gpu._embed_text_batch(all_texts)
            
            # 分离：前 N 个是文件 chunks，后面是视觉描述
            file_embeddings = embeddings[:len(texts)]
            desc_embeddings = embeddings[len(texts):] if desc_rows else []
            
            quantized = np.clip(np.round(file_embeddings[:, :EMBEDDING_DIM] * 127), -127, 127).astype(np.int8)
            
            # 按文件分组写入
            file_chunks = {}
            for i, chunk_data in enumerate(chunks_data):
                fid = chunk_data['file_id']
                if fid not in file_chunks:
                    file_chunks[fid] = []
                file_chunks[fid].append((chunk_data, quantized[i].tobytes()))
            
            wf, wc = 0, 0
            for fid, chunks in file_chunks.items():
                try:
                    c.execute("BEGIN TRANSACTION")
                    c.execute("INSERT OR REPLACE INTO rag_status (file_id, status) VALUES (?, 'processing')", (fid,))
                    c.execute("DELETE FROM rag_chunks WHERE file_id = ?", (fid,))
                    for chunk_data, vec_blob in chunks:
                        # 保存 chunk 文本前 500 字符到 meta（供 cross-encoder 重排序）
                        meta = dict(chunk_data['meta']) if chunk_data['meta'] else {}
                        meta['_text'] = chunk_data['text'][:500]
                        c.execute("INSERT INTO rag_chunks (file_id, chunk_index, meta_json, vector) VALUES (?, ?, ?, ?)",
                                  (fid, chunk_data['chunk_index'], 
                                   json.dumps(meta, ensure_ascii=False),
                                   vec_blob))
                        wc += 1
                    c.execute("UPDATE rag_status SET status='done' WHERE file_id=?", (fid,))
                    conn.commit()
                    wf += 1
                except Exception as e:
                    log.error(f"DB write error for file {fid}: {e}")
                    conn.rollback()
            
            # 嵌入视觉描述
            desc_embedded = 0
            if desc_rows and len(desc_embeddings) == len(desc_rows):
                desc_quantized = np.clip(np.round(desc_embeddings[:, :EMBEDDING_DIM] * 127), -127, 127).astype(np.int8)
                try:
                    c.execute("BEGIN TRANSACTION")
                    for idx, (desc_id, file_id, description) in enumerate(desc_rows):
                        meta = json.dumps({
                            'source': 'vision_description',
                            'description': description[:200] + '...' if len(description) > 200 else description
                        }, ensure_ascii=False)
                        vec_blob = desc_quantized[idx].tobytes()
                        c.execute("""
                            INSERT OR REPLACE INTO rag_chunks 
                            (file_id, chunk_index, meta_json, vector)
                            VALUES (?, -1, ?, ?)
                        """, (file_id, meta, vec_blob))
                        c.execute("UPDATE rag_descriptions SET embedded = 1 WHERE id = ?", (desc_id,))
                        desc_embedded += 1
                    conn.commit()
                    if desc_embedded > 0:
                        log.info(f"[Vision] Embedded {desc_embedded} descriptions")
                except Exception as e:
                    log.error(f"Description embed error: {e}")
                    conn.rollback()
            
            # 显式释放嵌入结果，防止内存泄漏
            del file_embeddings, desc_embeddings, quantized, file_chunks, embeddings, all_texts, texts
            gc.collect()
            
            return wf, wc + desc_embedded
        except Exception as e:
            log.error(f"Embedding batch error: {e}")
            # vLLM 超时时自动重启服务
            if 'timeout' in str(e).lower() or 'timed out' in str(e).lower():
                log.warning("vLLM timed out, restarting service...")
                stop_vllm_service()
                time.sleep(2)
                if start_vllm_service():
                    # 重新初始化 embedder 连接
                    gpu.load_model()
                    log.info("vLLM restarted, retrying...")
                else:
                    log.error("Failed to restart vLLM")
            return 0, 0
    
    batch_count = 0
    all_batches = [(batch, i, skip_ids, args.audio_threads) for i, batch in enumerate(file_batches)]
    
    for batch_result in pool.imap_unordered(_process_batch_wrapper, all_batches, chunksize=1):
        producer_id, stats, chunks_data = batch_result
        batch_count += 1
        total_stats['extracted'] += stats['extracted']
        total_stats['chunks'] += stats['chunks']
        total_stats['errors'] += stats['errors']
        total_stats['skipped'] += stats['skipped']
        total_stats['chars'] += stats['chars']
        
        # 立刻 embed + 写 DB
        wf, wc = embed_and_write_batch(chunks_data)
        written_files += wf
        written_chunks += wc
        
        # 显式释放本批次数据
        del chunks_data, batch_result
        
        # 定期汇报进度
        if batch_count % 50 == 0:
            elapsed = time.time() - t0
            rate = batch_count / elapsed if elapsed > 0 else 0
            pct = batch_count / len(file_batches) * 100
            log.info(f"\ud83d\udcca {batch_count:,}/{len(file_batches):,} batches ({pct:.1f}%) | {total_stats['skipped']:,} skipped | {written_chunks:,} chunks | {rate:.1f} batches/s")
        
        if written_files % 100 == 0 and written_files > 0:
            elapsed = time.time() - t0
            rate = written_files / elapsed if elapsed > 0 else 0
            pct = written_files / len(file_rows) * 100
            log.info(f"\ud83d\udcca {written_files:,}/{len(file_rows):,} files ({pct:.1f}%) | {written_chunks:,} chunks | {rate:.1f} files/s")
    
    pool.close()
    pool.join()
    conn.close()
    
    # Stop describer threads first (they may still be processing)
    log.info("Stopping describer threads...")
    stop_describers(describer_threads, timeout=120)
    
    # Stop services
    stop_ocr_service()
    stop_vllm_service()
    
    elapsed = time.time() - t0
    rate = written_chunks / elapsed if elapsed > 0 else 0
    
    log.info("\n" + "=" * 60)
    log.info("📊 PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"Stats: {total_stats}")
    log.info(f"Written: {written_files} files, {written_chunks} chunks")
    log.info(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    log.info(f"Rate: {rate:.1f} chunks/s")
    
    # Final count
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT file_id), COUNT(*) FROM rag_chunks")
    total_files_db, total_chunks = c.fetchone()
    c.execute("SELECT COUNT(*) FROM rag_status WHERE status='done'")
    done_files = c.fetchone()[0]
    conn.close()
    log.info(f"✅ Final: {total_files_db} files in rag_chunks, {done_files} marked done, {total_chunks} chunks")


# ============================================================
# 子命令入口
# ============================================================

def run_scan():
    """运行文件扫描"""
    log.info("🚀 Starting file scan...")
    from file_scanner import sync
    sync()
    log.info("✅ File scan complete")


def run_extract_meta():
    """运行元数据提取"""
    log.info("🚀 Starting metadata extraction...")
    from metadata_extractor import extract_metadata
    extract_metadata()
    log.info("✅ Metadata extraction complete")


def run_search(query: str, top_k: int = 50):
    """搜索数据库"""
    if not query:
        log.error("搜索关键词不能为空")
        return
    
    log.info(f"🔍 Searching for: {query}")
    try:
        from jina_v5_embedding import JinaV5Embedder
        embedder = JinaV5Embedder()
        query_vec = embedder.encode([query], task='retrieval', prompt_name='query')[0]
        query_vec = query_vec[:EMBEDDING_DIM]
        
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT file_id, chunk_index, meta_json, vector, f.path FROM rag_chunks r JOIN files f ON r.file_id = f.id WHERE f.is_deleted = 0")
        rows = c.fetchall()
        
        results = []
        for row in rows:
            vec_blob = row[3]
            if not vec_blob:
                continue
            arr = np.frombuffer(vec_blob, dtype=np.int8).astype(np.float32) / 127.0
            score = np.dot(query_vec, arr) / (np.linalg.norm(query_vec) * np.linalg.norm(arr))
            results.append((score, row[4], row[2]))
        
        results.sort(reverse=True, key=lambda x: x[0])
        results = results[:top_k]
        
        log.info(f"找到 {len(results)} 个结果:")
        for i, (score, path, meta) in enumerate(results[:10]):
            log.info(f"  {i+1}. [{score:.3f}] {path}")
            if meta:
                try:
                    m = json.loads(meta)
                    if '_text' in m:
                        log.info(f"      {m['_text'][:100]}...")
                except Exception:
                    pass
        
        conn.close()
        
    except Exception as e:
        log.error(f"搜索失败: {e}")


def run_serve():
    """启动 FastAPI Web 服务"""
    log.info("🚀 Starting FastAPI server...")
    from server import main
    main()


def main_entry():
    """子命令入口"""
    import sys
    
    # 显示子命令帮助
    if '-h' in sys.argv or '--help' in sys.argv:
        if len(sys.argv) == 2:
            # 只有 --help，显示子命令帮助
            print("用法: python run_pipeline.py <子命令> [选项]")
            print("")
            print("子命令:")
            print("  scan              扫描文件并同步到数据库")
            print("  extract-meta      提取媒体元数据（图片/视频/音频/PDF）")
            print("  process           RAG 处理（嵌入搜索向量）")
            print("  search <关键词>   搜索数据库")
            print("  serve             启动 FastAPI Web 服务")
            print("")
            print("旧版参数兼容（直接运行 process）:")
            print("  --force           强制重新处理所有文件")
            print("  --resume          跳过已处理的文件")
            print("  --limit N         只处理 N 个文件")
            print("  --types 类型      只处理指定类型文件")
            print("")
            print("示例:")
            print("  python run_pipeline.py scan")
            print("  python run_pipeline.py extract-meta")
            print("  python run_pipeline.py process --force")
            print("  python run_pipeline.py search \"文档管理\"")
            print("  python run_pipeline.py serve")
            print("")
            print("详细帮助: python run_pipeline.py process --help")
            return
    
    # 解析子命令
    if len(sys.argv) > 1:
        subcommand = sys.argv[1]
        
        if subcommand == 'scan':
            # 文件扫描
            run_scan()
            return
        
        elif subcommand == 'extract-meta':
            # 元数据提取
            run_extract_meta()
            return
        
        elif subcommand == 'process':
            # RAG 处理（旧版参数兼容）
            # 移除 'process' 子命令，传递剩余参数给 run_rag_process
            old_argv = sys.argv
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            run_rag_process()
            return
        
        elif subcommand == 'search':
            # 搜索
            if len(sys.argv) < 3:
                print("用法: python run_pipeline.py search <关键词> [--top-k N]")
                sys.exit(1)
            query = sys.argv[2]
            top_k = 50
            for i in range(3, len(sys.argv)):
                if sys.argv[i] == '--top-k' and i + 1 < len(sys.argv):
                    top_k = int(sys.argv[i + 1])
            run_search(query, top_k)
            return
        
        elif subcommand == 'serve':
            # 启动 Web 服务
            run_serve()
            return
        
        elif subcommand.startswith('-') or subcommand.startswith('--'):
            # 旧版参数（直接运行 process）
            # 如 --force, --resume, --limit 等
            run_rag_process()
            return
        
        else:
            print(f"未知子命令: {subcommand}")
            print("可用子命令: scan, extract-meta, process, search, serve")
            sys.exit(1)
    
    # 无参数或直接运行 → 默认 process
    run_rag_process()


if __name__ == "__main__":
    main_entry()