#!/usr/bin/env python3
"""描述嵌入器 — 消费 rag_descriptions 表，调用 Jina 生成向量

独立进程，统一处理所有视觉描述和文本的嵌入：
1. 从 rag_descriptions 表读取未嵌入的描述
2. 批量调用 Jina 嵌入模型
3. 写入 rag_chunks 表
4. 标记 rag_descriptions.embedded = 1

这样避免多个进程各自加载 Jina，节省显存。

配置从 config.yaml 加载。
"""

import os
import sys
import json
import sqlite3
import time
import logging
import threading
import numpy as np
from typing import List, Tuple

# Import config loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_loader import (
        get_db_path, get_log_dir, get_embedding_dim, get_config
    )
    config = get_config()
    DB_PATH = get_db_path()
    LOG_DIR = get_log_dir()
    EMBEDDING_DIM = get_embedding_dim()
    BATCH_SIZE = config['embedding']['description_batch_size']
except ImportError:
    # Fallback for standalone usage
    DB_PATH = os.environ.get("DB_PATH", "./file_index.db")
    LOG_DIR = os.environ.get("LOG_DIR", "./logs")
    EMBEDDING_DIM = 512
    BATCH_SIZE = 8  # 每批处理 8 个描述

# 日志
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'description_embedder.log')),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# 线程锁
db_lock = threading.Lock()

# Jina 嵌入器（全局单例）
_jina_embedder = None

def get_jina_embedder():
    """获取或创建 Jina 嵌入器（单例）"""
    global _jina_embedder
    if _jina_embedder is None:
        from jina_v5_embedding import JinaV5Embedder
        _jina_embedder = JinaV5Embedder()
        log.info("✅ Jina embedder loaded")
    return _jina_embedder

def get_db():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH, timeout=30)

def get_pending_descriptions(limit: int = BATCH_SIZE) -> List[Tuple[int, int, str, str]]:
    """从数据库获取未嵌入的描述（排除失败标记）
    
    Returns: [(id, file_id, description, source), ...]
    """
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, file_id, description, source
        FROM rag_descriptions
        WHERE embedded = 0
        AND source NOT LIKE '%_failed'
        ORDER BY created_at ASC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def embed_texts(texts: List[str]) -> np.ndarray:
    """调用 Jina 嵌入模型生成向量"""
    embedder = get_jina_embedder()
    return embedder.encode(texts)

def quantize_embedding(embedding: np.ndarray) -> bytes:
    """量化嵌入为 int8（先 L2 归一化再量化）"""
    # L2 归一化（量化前必须做，否则余弦相似度不准确）
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    quantized = np.clip(np.round(embedding[:EMBEDDING_DIM] * 127), -127, 127).astype(np.int8)
    return quantized.tobytes()

def process_batch(descriptions: List[Tuple[int, int, str, str]]) -> int:
    """处理一批描述：嵌入并写入 rag_chunks
    
    Returns: 成功处理的数量
    """
    if not descriptions:
        return 0
    
    ids = [d[0] for d in descriptions]
    file_ids = [d[1] for d in descriptions]
    texts = [d[2] for d in descriptions]
    sources = [d[3] for d in descriptions]
    
    # 批量嵌入
    try:
        log.info(f"[Jina] Embedding {len(texts)} descriptions...")
        embeddings = embed_texts(texts)
        log.info(f"[Jina] Embedded {len(texts)} descriptions")
    except Exception as e:
        log.error(f"Embedding failed: {e}")
        return 0
    
    # 写入数据库
    success_count = 0
    try:
        with db_lock:
            conn = get_db()
            c = conn.cursor()
            
            for i, (desc_id, file_id, description, source) in enumerate(descriptions):
                try:
                    # 量化嵌入
                    vector_bytes = quantize_embedding(embeddings[i])
                    
                    # 构建 meta
                    meta = json.dumps({
                        'source': f'vision_{source}',
                        'description': description[:200] + '...' if len(description) > 200 else description
                    })
                    
                    # 插入 rag_chunks（用 chunk_index=-1 区分视觉描述，避免覆盖 pipeline 的 chunk_index=0）
                    c.execute("""
                        INSERT OR REPLACE INTO rag_chunks 
                        (file_id, chunk_index, meta_json, vector, created_at)
                        VALUES (?, -1, ?, ?, datetime('now'))
                    """, (file_id, meta, vector_bytes))
                    
                    # 标记为已嵌入
                    c.execute("""
                        UPDATE rag_descriptions
                        SET embedded = 1
                        WHERE id = ?
                    """, (desc_id,))
                    
                    success_count += 1
                    log.info(f"✅ Embedded file_id={file_id}: {len(description)} chars")
                    
                except Exception as e:
                    log.error(f"Failed to embed desc_id={desc_id}: {e}")
            
            conn.commit()
            conn.close()
    except Exception as e:
        log.error(f"DB write failed: {e}")
    
    return success_count

def main():
    """主循环"""
    log.info("🚀 Description Embedder started")
    log.info(f"Batch size: {BATCH_SIZE}")
    log.info(f"DB path: {DB_PATH}")
    
    # 预加载 Jina 模型
    get_jina_embedder()
    
    total_processed = 0
    total_failed = 0
    
    while True:
        # 获取待嵌入的描述
        descriptions = get_pending_descriptions(limit=BATCH_SIZE)
        
        if not descriptions:
            log.info("📭 No pending descriptions. Sleeping 30s...")
            time.sleep(30)
            continue
        
        log.info(f"📥 Found {len(descriptions)} pending descriptions")
        
        # 处理批次
        success = process_batch(descriptions)
        failed = len(descriptions) - success
        
        total_processed += success
        total_failed += failed
        
        log.info(f"📊 Batch done: {success} success, {failed} failed | Total: {total_processed} processed, {total_failed} failed")
        
        # 短暂休眠
        time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Description Embedder")
    parser.add_argument("--config", type=str, default="", help="Path to config.yaml")
    args = parser.parse_args()
    
    if args.config:
        from config_loader import reset_config_cache, load_config
        reset_config_cache()
        load_config(args.config)
    
    main()