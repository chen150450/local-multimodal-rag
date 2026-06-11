#!/usr/bin/env python3
"""
Jina Embeddings v5 Text Small — vLLM API Client

通过 vLLM 提供的 OpenAI-compatible API 调用 embedding 服务。
vLLM 服务需要预先启动：
    vllm serve /path/to/jina-embeddings-v5-text-small/ \
        --port 8001 --dtype bfloat16 --trust-remote-code \
        --hf-overrides '{"architectures": ["TransformersEmbeddingModel"]}'

Usage:
    from jina_v5_embedding import JinaV5Embedder
    
    embedder = JinaV5Embedder()
    vectors = embedder.encode(["Hello world"], task="retrieval", prompt_name="document")
    print(vectors.shape)  # (1, 512)

Configuration loaded from config.yaml via config_loader.
"""

import os
import logging
import numpy as np
import fcntl
import time
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import config loader
try:
    from config_loader import (
        get_embedding_dim, get_vllm_api_url, get_vllm_model_name,
        get_jina_model_path, get_config, get_env
    )
    config = get_config()
    # Default dimension: 512 (Matryoshka truncation). Full model outputs 1024.
    EMBEDDING_DIM = get_embedding_dim()
    VLLM_API_URL = get_vllm_api_url()
    VLLM_MODEL_NAME = get_vllm_model_name()
except ImportError:
    # Fallback for standalone usage
    EMBEDDING_DIM = 512
    VLLM_API_URL = os.environ.get("VLLM_API_URL", "http://localhost:8001")
    VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "")

# 全局文件锁路径（跨进程序列化嵌入操作）
JINA_LOCK_PATH = "/tmp/jina_embed.lock"


class JinaV5Embedder:
    """Jina Embeddings v5 via vLLM API."""
    
    def __init__(self, model_path: str = None, device: str = "cuda",
                 max_length: int = 32768, batch_size: int = 32,
                 truncate_dim: int = None,
                 api_url: str = None,
                 model_name: str = None):
        """
        Args:
            model_path: (deprecated, ignored) kept for API compatibility
            device: (deprecated, ignored) vLLM handles device
            max_length: max token length (model supports up to 32768)
            batch_size: default batch size for encode
            truncate_dim: Matryoshka truncation dimension (1024/512/256/...)
            api_url: vLLM API base URL
            model_name: model name in vLLM
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim or EMBEDDING_DIM
        self.api_url = api_url or VLLM_API_URL
        self.model_name = model_name or VLLM_MODEL_NAME
        self._checked = False
    
    def _check_service(self):
        """Check if vLLM service is available."""
        if self._checked:
            return
        
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"✅ vLLM service available at {self.api_url}")
                self._checked = True
            else:
                raise RuntimeError(f"vLLM health check failed: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to vLLM at {self.api_url}. "
                f"Please start vLLM service first:\n"
                f"  vllm serve /path/to/jina-embeddings-v5-text-small/ \\\n"
                f"      --port 8001 --dtype bfloat16 --trust-remote-code \\\n"
                f"      --hf-overrides '{{\"architectures\": [\"TransformersEmbeddingModel\"]}}'"
            )
    
    def _health_check(self):
        """Quick health check before each API call. Raises on failure."""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=2)
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM unhealthy: status {resp.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM health check failed: {e}")
    
    def encode(self, texts: List[str], task: str = "retrieval",
               prompt_name: str = "document",
               batch_size: Optional[int] = None,
               max_retries: int = 3) -> np.ndarray:
        """Encode texts to embeddings via vLLM API.
        
        Args:
            texts: list of text strings
            task: 'retrieval', 'text-matching', 'clustering', 'classification'
            prompt_name: 'query' or 'document' (for retrieval task)
            batch_size: override default batch size
            max_retries: max retry count on failure
        
        Returns:
            numpy array of shape (len(texts), truncate_dim), L2-normalized
        """
        self._check_service()
        
        bs = batch_size or self.batch_size
        all_vecs = []
        
        # 获取跨进程文件锁（确保同一时刻只有一个进程在嵌入）
        lock_fd = open(JINA_LOCK_PATH, 'w')
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                
                # 注意：vLLM 无法正确处理 Jina v5 的 PeftMixedModel adapter 路由，
                # 加 "Query:"/"Document:" 前缀会导致所有文本返回相同向量。
                # 因此不加速度前缀，直接 encode 原文。
                # query vs document 的区分在调用侧（search_agg.py）处理。
                prefixed_batch = batch  # 不加前缀
                
                logger.info(f"[Jina] Batch {i//bs+1}: {len(batch)} texts via API (no prefix)")
                
                # 带重试的嵌入
                vecs = None
                for attempt in range(max_retries):
                    try:
                        # 检查 vLLM 健康状态（避免发请求到已卡死的服务）
                        self._health_check()
                        vecs = self._call_api(prefixed_batch)
                        break
                    except Exception as e:
                        logger.error(f"[Jina] API error (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            self._checked = False  # 重置健康检查状态
                        else:
                            raise
                
                # 批次间间隔，避免连续请求压垮 vLLM EngineCore
                if i + bs < len(texts):
                    time.sleep(0.2)
                
                all_vecs.append(vecs)
        
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
        
        result = np.concatenate(all_vecs, axis=0) if len(all_vecs) > 1 else all_vecs[0]
        
        # Matryoshka 截断
        if result.shape[1] > self.truncate_dim:
            result = result[:, :self.truncate_dim]
        
        # Ensure L2-normalized
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms
        
        return result
    
    def _call_api(self, texts: List[str]) -> np.ndarray:
        """Call vLLM embedding API."""
        # 过滤脏数据：包含大量 null bytes 的二进制文件
        clean_texts = []
        for text in texts:
            # 跳过包含大量 null bytes 的文本（二进制文件污染）
            if text.count('\x00') > 10 or text.count('\u0000') > 10:
                logger.warning(f"[Jina] Skipping binary-contaminated text ({len(text)} chars)")
                clean_texts.append(" ")  # 空格占位，vLLM 拒绝空字符串
            elif not text or not text.strip():
                clean_texts.append(" ")  # 空文本也用空格占位
            else:
                clean_texts.append(text)
        
        # 截断超长文本（vLLM max-model-len=16384 tokens）
        # chunker_v2 MAX_CHARS=12000：12000 中文字 ≈ 8000 tokens，安全在 16384 以内
        MAX_CHARS = 12000
        truncated_texts = [text[:MAX_CHARS] if len(text) > MAX_CHARS else text for text in clean_texts]
        
        resp = requests.post(
            f"{self.api_url}/v1/embeddings",
            json={
                "model": self.model_name,
                "input": truncated_texts,
            },
            timeout=120,
            proxies={"http": None, "https": None}  # 绕过代理，直连 localhost
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
        
        data = resp.json()
        
        # 提取 embeddings（按 index 排序）
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        vecs = np.array([item["embedding"] for item in embeddings], dtype=np.float32)
        
        logger.info(f"[Jina] Got {vecs.shape[0]} embeddings, dim={vecs.shape[1]}")
        
        return vecs
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (after truncation)."""
        return self.truncate_dim


# 兼容旧代码的函数
def load_model(model_path: str = None, **kwargs) -> JinaV5Embedder:
    """Load JinaV5Embedder (API client, no local model loading)."""
    return JinaV5Embedder(model_path=model_path, **kwargs)