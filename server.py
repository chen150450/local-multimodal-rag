#!/usr/bin/env python3
"""
FastAPI server for Local Multimodal RAG
- REST API for search
- Config management API
- Pipeline management API
- Static file serving for web UI

Usage:
  python server.py
  python run_pipeline.py serve
"""

import os
import sys
import json
import sqlite3
import traceback
import asyncio
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 配置加载
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_loader import get_config, get_env, reset_config_cache, load_config
    config = get_config()
    DB_PATH = config.get('database', {}).get('path', './file_index.db')
    EMBEDDING_DIM = config.get('embedding', {}).get('dimension', 512)
    server_config = config.get('server', {})
    HOST = server_config.get('host', '0.0.0.0')
    PORT = server_config.get('port', 8100)
    CORS_ORIGINS = server_config.get('cors_origins', ['*'])
    search_config = config.get('search', {})
    DEFAULT_TOP_K = search_config.get('default_top_k', 50)
    MAX_TOP_K = search_config.get('max_top_k', 200)
except ImportError:
    DB_PATH = './file_index.db'
    EMBEDDING_DIM = 512
    HOST = '0.0.0.0'
    PORT = 8100
    CORS_ORIGINS = ['*']
    DEFAULT_TOP_K = 50
    MAX_TOP_K = 200

# 端口环境变量覆盖
PORT = int(os.environ.get('RAG_PORT', PORT))

# 数据目录
DATA_DIR = Path(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else 'data')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 日志目录
LOG_DIR = Path(config.get('logging', {}).get('dir', './logs'))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 静态文件目录
STATIC_DIR = Path(__file__).parent / 'static'
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Pydantic Models
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    file_types: Optional[List[str]] = None
    path_filter: Optional[str] = None


class SearchResult(BaseModel):
    path: str
    score: float
    snippet: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    file_id: Optional[int] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any]


class PipelineStatus(BaseModel):
    scanner: Dict[str, Any]
    extractor: Dict[str, Any]
    embedder: Dict[str, Any]
    ocr: Dict[str, Any]
    vision: Dict[str, Any]


class StatsResponse(BaseModel):
    total_files: int
    total_chunks: int
    files_with_chunks: int
    by_type: Dict[str, int]
    by_ext: Dict[str, int]
    db_size_mb: float


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Local Multimodal RAG API",
    description="REST API for local multimodal RAG search and management",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================
# Database Helper
# ============================================================

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def int8_to_float(blob: bytes) -> np.ndarray:
    """int8 quantized vector → float32"""
    arr = np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# ============================================================
# 搜索 API
# ============================================================

@app.post("/api/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """POST 搜索 API"""
    return await _do_search(request.query, request.top_k, request.file_types, request.path_filter)


@app.get("/api/search")
async def search_get(q: str, top_k: int = DEFAULT_TOP_K, file_types: Optional[str] = None, path: Optional[str] = None):
    """GET 搜索 API"""
    types_list = [t.strip() for t in file_types.split(',') if t.strip()] if file_types else None
    return await _do_search(q, top_k, types_list, path)


async def _do_search(query: str, top_k: int, file_types: Optional[List[str]] = None, path_filter: Optional[str] = None) -> SearchResponse:
    """执行搜索"""
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    top_k = min(top_k, MAX_TOP_K)
    
    try:
        # 获取查询向量（通过嵌入服务）
        from jina_v5_embedding import JinaV5Embedder
        embedder = JinaV5Embedder()
        query_vec = embedder.encode([query], task='retrieval', prompt_name='query')[0]
        query_vec = query_vec[:EMBEDDING_DIM]
        
        # 连接数据库
        conn = get_db()
        c = conn.cursor()
        
        # 构建查询
        sql = """
            SELECT r.file_id, r.chunk_index, r.meta_json, r.vector, f.path, f.name, f.ext, f.mime_type, f.media_meta
            FROM rag_chunks r
            JOIN files f ON r.file_id = f.id
            WHERE f.is_deleted = 0
        """
        params = []
        
        # 文件类型过滤
        if file_types:
            exts = []
            for ft in file_types:
                if ft == 'pdf':
                    exts.extend(['.pdf'])
                elif ft == 'image':
                    exts.extend(['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif'])
                elif ft == 'video':
                    exts.extend(['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'])
                elif ft == 'audio':
                    exts.extend(['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus'])
                elif ft == 'text':
                    exts.extend(['.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.log', '.cfg', '.ini', '.conf'])
                elif ft == 'code':
                    exts.extend(['.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.h', '.hpp', '.java', '.kt', '.rs', '.go', '.rb', '.php'])
                else:
                    exts.append(f'.{ft}')
            if exts:
                placeholders = ','.join(['?'] * len(exts))
                sql += f" AND LOWER(f.ext) IN ({placeholders})"
                params.extend(exts)
        
        # 路径过滤
        if path_filter:
            sql += " AND f.path LIKE ?"
            params.append(f"{path_filter}%")
        
        c.execute(sql, params)
        rows = c.fetchall()
        
        # 计算相似度
        results = []
        for row in rows:
            vec_blob = row['vector']
            if not vec_blob:
                continue
            chunk_vec = int8_to_float(vec_blob)
            score = cosine_similarity(query_vec, chunk_vec)
            
            # 解析 meta
            meta = {}
            try:
                if row['meta_json']:
                    meta = json.loads(row['meta_json'])
            except Exception:
                pass
            
            # 获取 snippet
            snippet = meta.get('_text', '')
            if not snippet:
                # 从文件元数据获取
                try:
                    if row['media_meta']:
                        media_meta = json.loads(row['media_meta'])
                        snippet = media_meta.get('description', '')
                except Exception:
                    pass
            
            results.append(SearchResult(
                path=row['path'],
                score=float(score),
                snippet=snippet[:500] if snippet else None,
                meta={
                    'ext': row['ext'],
                    'mime_type': row['mime_type'],
                    'chunk_index': row['chunk_index'],
                    'source': meta.get('source', 'unknown'),
                },
                file_id=row['file_id']
            ))
        
        # 排序并取 top_k
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        conn.close()
        
        return SearchResponse(
            results=results,
            total=len(results),
            query=query
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ============================================================
# 配置管理 API
# ============================================================

@app.get("/api/config")
async def get_config_api():
    """获取当前配置（API keys 脱敏）"""
    try:
        cfg = get_config()
        # 脱敏处理
        safe_cfg = {}
        for key, value in cfg.items():
            if isinstance(value, dict):
                safe_cfg[key] = {}
                for k, v in value.items():
                    if 'api_key' in k.lower() or 'key' in k.lower():
                        safe_cfg[key][k] = "***REDACTED***" if v else None
                    elif isinstance(v, dict):
                        safe_cfg[key][k] = {}
                        for kk, vv in v.items():
                            if 'api_key' in kk.lower() or 'key' in kk.lower():
                                safe_cfg[key][k][kk] = "***REDACTED***" if vv else None
                            else:
                                safe_cfg[key][k][kk] = vv
                    else:
                        safe_cfg[key][k] = v
            else:
                safe_cfg[key] = value
        
        # 添加环境变量状态
        safe_cfg['env_status'] = {
            'QWEN_API_KEY': bool(get_env('QWEN_API_KEY')),
            'LMSTUDIO_URL': bool(get_env('LMSTUDIO_URL')),
        }
        
        return safe_cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")


@app.put("/api/config")
async def update_config_api(request: ConfigUpdateRequest):
    """更新配置（部分更新）"""
    try:
        # 读取现有配置
        config_path = Path(__file__).parent / 'config.yaml'
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="config.yaml not found")
        
        import yaml
        with open(config_path, 'r') as f:
            current_cfg = yaml.safe_load(f) or {}
        
        # 合并更新
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(current_cfg, request.config)
        
        # 写回文件
        with open(config_path, 'w') as f:
            yaml.dump(current_cfg, f, default_flow_style=False, allow_unicode=True)
        
        # 重置缓存
        reset_config_cache()
        
        return {"status": "updated", "message": "Configuration updated. Reload to apply."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config update error: {str(e)}")


@app.post("/api/config/reload")
async def reload_config():
    """重新加载配置"""
    try:
        reset_config_cache()
        load_config()
        return {"status": "reloaded", "message": "Configuration reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config reload error: {str(e)}")


@app.get("/api/config/status")
async def get_config_status():
    """返回各组件状态"""
    try:
        status = {
            "database": {"connected": False, "path": DB_PATH},
            "embedding": {"available": False, "dimension": EMBEDDING_DIM},
            "ocr": {"available": False},
            "vision": {"available": False, "endpoints": []},
        }
        
        # 检查数据库
        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM files")
            status["database"]["connected"] = True
            status["database"]["file_count"] = c.fetchone()[0]
            conn.close()
        except Exception:
            pass
        
        # 检查嵌入服务
        try:
            import requests
            vllm_url = config.get('embedding', {}).get('vllm', {}).get('api_url', 'http://localhost:8001')
            resp = requests.get(f"{vllm_url}/health", timeout=2)
            status["embedding"]["available"] = resp.status_code == 200
        except Exception:
            pass
        
        # 检查 OCR
        try:
            import requests
            ocr_url = config.get('ocr', {}).get('api_url', 'http://127.0.0.1:8002')
            resp = requests.get(f"{ocr_url}/health", timeout=2, proxies={"http": None, "https": None})
            status["ocr"]["available"] = resp.status_code == 200
        except Exception:
            pass
        
        # 检查视觉模型
        vision_cfg = config.get('vision', {})
        endpoints = vision_cfg.get('endpoints', [])
        if not endpoints:
            # 旧配置格式兼容
            endpoints = [{
                'api_base': vision_cfg.get('lmstudio', {}).get('url', 'http://localhost:1234/v1'),
                'model': vision_cfg.get('lmstudio', {}).get('model', 'default')
            }]
        
        for ep in endpoints:
            try:
                import requests
                base = ep.get('api_base', '')
                if base:
                    resp = requests.get(f"{base.rstrip('/')}/models", timeout=2)
                    status["vision"]["endpoints"].append({
                        "url": base,
                        "available": resp.status_code == 200,
                        "model": ep.get('model', 'unknown')
                    })
            except Exception:
                status["vision"]["endpoints"].append({
                    "url": ep.get('api_base', ''),
                    "available": False,
                    "model": ep.get('model', 'unknown')
                })
        
        status["vision"]["available"] = any(ep.get("available", False) for ep in status["vision"]["endpoints"])
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check error: {str(e)}")


# ============================================================
# 管线管理 API
# ============================================================

pipeline_status = {
    "scanner": {"running": False, "last_run": None},
    "extractor": {"running": False, "last_run": None},
    "processor": {"running": False, "last_run": None},
}

@app.post("/api/pipeline/scan")
async def trigger_scan():
    """触发文件扫描"""
    try:
        if pipeline_status["scanner"]["running"]:
            raise HTTPException(status_code=409, detail="Scanner already running")
        
        # 后台执行扫描
        def run_scan():
            pipeline_status["scanner"]["running"] = True
            try:
                from file_scanner import sync
                sync()
            except Exception as e:
                traceback.print_exc()
            finally:
                pipeline_status["scanner"]["running"] = False
                pipeline_status["scanner"]["last_run"] = datetime.now().isoformat()
        
        thread = threading.Thread(target=run_scan, daemon=True)
        thread.start()
        
        return {"status": "started", "message": "File scan started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan error: {str(e)}")


@app.post("/api/pipeline/extract")
async def trigger_extract():
    """触发元数据提取"""
    try:
        if pipeline_status["extractor"]["running"]:
            raise HTTPException(status_code=409, detail="Extractor already running")
        
        def run_extract():
            pipeline_status["extractor"]["running"] = True
            try:
                from metadata_extractor import extract_metadata
                extract_metadata()
            except Exception as e:
                traceback.print_exc()
            finally:
                pipeline_status["extractor"]["running"] = False
                pipeline_status["extractor"]["last_run"] = datetime.now().isoformat()
        
        thread = threading.Thread(target=run_extract, daemon=True)
        thread.start()
        
        return {"status": "started", "message": "Metadata extraction started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extract error: {str(e)}")


@app.post("/api/pipeline/process")
async def trigger_process():
    """触发 RAG 处理"""
    try:
        if pipeline_status["processor"]["running"]:
            raise HTTPException(status_code=409, detail="Processor already running")
        
        def run_process():
            pipeline_status["processor"]["running"] = True
            try:
                import run_pipeline
                run_pipeline.main()
            except Exception as e:
                traceback.print_exc()
            finally:
                pipeline_status["processor"]["running"] = False
                pipeline_status["processor"]["last_run"] = datetime.now().isoformat()
        
        thread = threading.Thread(target=run_process, daemon=True)
        thread.start()
        
        return {"status": "started", "message": "RAG processing started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process error: {str(e)}")


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """获取管线状态"""
    return pipeline_status


# ============================================================
# 统计 API
# ============================================================

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取数据库统计"""
    try:
        conn = get_db()
        c = conn.cursor()
        
        # 总文件数
        c.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0")
        total_files = c.fetchone()[0]
        
        # 总 chunks
        c.execute("SELECT COUNT(*) FROM rag_chunks")
        total_chunks = c.fetchone()[0]
        
        # 有 chunks 的文件数
        c.execute("SELECT COUNT(DISTINCT file_id) FROM rag_chunks")
        files_with_chunks = c.fetchone()[0]
        
        # 按类型统计
        c.execute("""
            SELECT 
                CASE 
                    WHEN LOWER(ext) IN ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif') THEN 'image'
                    WHEN LOWER(ext) IN ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm') THEN 'video'
                    WHEN LOWER(ext) IN ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus') THEN 'audio'
                    WHEN LOWER(ext) = '.pdf' THEN 'pdf'
                    WHEN LOWER(ext) IN ('.txt', '.md', '.json', '.yaml', '.yml', '.toml', '.log', '.cfg', '.ini', '.conf') THEN 'text'
                    WHEN LOWER(ext) IN ('.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.h', '.hpp', '.java', '.kt', '.rs', '.go', '.rb', '.php') THEN 'code'
                    ELSE 'other'
                END as type,
                COUNT(*) as count
            FROM files WHERE is_deleted = 0
            GROUP BY type
        """)
        by_type = {row['type']: row['count'] for row in c.fetchall()}
        
        # 按扩展名统计（top 20）
        c.execute("""
            SELECT LOWER(ext) as ext, COUNT(*) as count
            FROM files WHERE is_deleted = 0 AND ext IS NOT NULL
            GROUP BY ext ORDER BY count DESC LIMIT 20
        """)
        by_ext = {row['ext']: row['count'] for row in c.fetchall()}
        
        # 数据库大小
        db_size = os.path.getsize(DB_PATH) / 1024 / 1024 if os.path.exists(DB_PATH) else 0
        
        conn.close()
        
        return StatsResponse(
            total_files=total_files,
            total_chunks=total_chunks,
            files_with_chunks=files_with_chunks,
            by_type=by_type,
            by_ext=by_ext,
            db_size_mb=db_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    try:
        conn = get_db()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception:
        return {"status": "unhealthy", "database": "disconnected"}


# ============================================================
# 静态页面
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页（Web UI）"""
    index_path = STATIC_DIR / 'index.html'
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    else:
        # 占位页面
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Local Multimodal RAG</title>
    <meta charset="utf-8">
    <style>
        body { font-family: system-ui; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .api-link { display: block; margin: 10px 0; padding: 15px; background: white; border-radius: 8px; }
        a { color: #0066cc; }
    </style>
</head>
<body>
    <h1>🚀 Local Multimodal RAG</h1>
    <p>FastAPI backend is running. Web UI will be added here.</p>
    <h2>API Endpoints</h2>
    <div class="api-link"><a href="/api/health">/api/health</a> - Health check</div>
    <div class="api-link"><a href="/api/config">/api/config</a> - Current configuration</div>
    <div class="api-link"><a href="/api/config/status">/api/config/status</a> - Component status</div>
    <div class="api-link"><a href="/api/stats">/api/stats</a> - Database statistics</div>
    <div class="api-link"><a href="/docs">/docs</a> - OpenAPI documentation</div>
</body>
</html>
        """, status_code=200)


# ============================================================
# 错误处理
# ============================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """通用错误处理"""
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "detail": traceback.format_exc()}
    )


# ============================================================
# 主入口
# ============================================================

def main():
    """启动服务器"""
    print(f"🚀 Starting Local Multimodal RAG server on {HOST}:{PORT}")
    print(f"📊 API docs: http://{HOST}:{PORT}/docs")
    print(f"🔍 Web UI: http://{HOST}:{PORT}/")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()