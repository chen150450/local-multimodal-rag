#!/usr/bin/env python3
"""视觉描述生成器 — 统一 OpenAI 兼容 API

处理 rag_ocr_status 中 OCR 失败的文件：
1. 从数据库读取 OCR 失败的文件
2. 根据文件类型渲染为图片（PDF/Office/djvu 需要先渲染）
3. 调用 OpenAI 兼容视觉模型生成描述
4. 写入 rag_descriptions 表（由 pipeline 统一嵌入）

支持多个端点 fallback：
- LMStudio / Ollama / vLLM / Qwen 云 API / 任何 OpenAI 兼容端点

配置从 config.yaml 和 .env 加载。
"""

import os
import sys
import json
import sqlite3
import base64
import time
import logging
import io
import subprocess
import tempfile
import threading
import platform
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import gc

# Import config loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_loader import (
        get_db_path, get_log_dir, get_config, get_env,
        get_limits_config, get_pipeline_config
    )
    config = get_config()
    DB_PATH = get_db_path()
    LOG_DIR = get_log_dir()
    limits_config = get_limits_config()
    pipeline_config = get_pipeline_config()
    
    # 视觉模型配置（OpenAI 兼容）
    vision_config = config.get('vision', {})
    
    # 支持新旧两种配置格式
    # 新格式：vision.endpoints (列表)
    # 旧格式：vision.qwen / vision.lmstudio (兼容)
    if 'endpoints' in vision_config:
        # 新格式
        ENDPOINTS = vision_config['endpoints']
    else:
        # 旧格式兼容
        ENDPOINTS = []
        # Qwen (云端)
        qwen_cfg = vision_config.get('qwen', {})
        if qwen_cfg:
            ENDPOINTS.append({
                'api_base': qwen_cfg.get('api_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                'api_key': get_env('QWEN_API_KEY', ''),
                'model': qwen_cfg.get('model', 'qwen-vl-max'),
                'max_tokens': qwen_cfg.get('max_tokens', 2048),
                'timeout': qwen_cfg.get('timeout', 60),
                'name': 'qwen_cloud',
                'rate_limit': qwen_cfg.get('min_interval', 60),
            })
        # LMStudio (本地)
        lmstudio_cfg = vision_config.get('lmstudio', {})
        if lmstudio_cfg:
            ENDPOINTS.append({
                'api_base': get_env('LMSTUDIO_URL', lmstudio_cfg.get('url', 'http://localhost:1234/v1')),
                'api_key': get_env('LMSTUDIO_API_KEY', ''),
                'model': lmstudio_cfg.get('model', 'default'),
                'max_tokens': lmstudio_cfg.get('max_tokens', 2048),
                'timeout': 120,
                'name': 'lmstudio',
                'max_parallel': lmstudio_cfg.get('max_parallel', 2),
            })
    
    # 全局配置
    MAX_WORKERS = vision_config.get('max_workers', 2)
    MAX_RETRIES = vision_config.get('max_retries', 3)
    MAX_IMAGE_FILE_BYTES = limits_config.get('max_image_bytes', 100 * 1024 * 1024)
    MAX_DESCRIBE_PAGES = limits_config.get('max_describe_pages', 500)
    FETCH_LIMIT = pipeline_config.get('fetch_limit', 20)
    
except ImportError:
    # Fallback for standalone usage
    DB_PATH = os.path.expanduser("./data/file_index.db")
    LOG_DIR = os.path.expanduser("./logs")
    ENDPOINTS = [
        {
            'api_base': os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1"),
            'api_key': os.environ.get("LMSTUDIO_API_KEY", ""),
            'model': "qwen2.5-vl-7b",
            'max_tokens': 2048,
            'timeout': 60,
            'name': 'lmstudio',
        }
    ]
    MAX_WORKERS = 2
    MAX_RETRIES = 3
    MAX_IMAGE_FILE_BYTES = 100 * 1024 * 1024
    MAX_DESCRIBE_PAGES = 500
    FETCH_LIMIT = 20

# 文件类型
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tga', '.tiff', '.tif'}
OFFICE_EXTS = {'.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}

# ============================================================
# Logger
# ============================================================
log = logging.getLogger("rag_pipeline.describer")

# ============================================================
# DB
# ============================================================
_db_lock = threading.Lock()

def _get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_ocr_failed_files(limit: int = FETCH_LIMIT) -> List[Tuple[int, str, str]]:
    """从数据库获取 OCR 失败且未达重试上限的文件。
    
    优先级：图片 > PDF > Office > djvu
    Returns: [(file_id, path, reason), ...]
    """
    conn = _get_db()
    c = conn.cursor()
    all_rows = []

    # 1. 图片
    c.execute("""
        SELECT r.file_id, f.path, r.reason
        FROM rag_ocr_status r
        JOIN files f ON r.file_id = f.id
        WHERE r.status = 'insufficient'
        AND NOT EXISTS (
            SELECT 1 FROM rag_descriptions rd 
            WHERE rd.file_id = r.file_id 
            AND (rd.source NOT IN ('vision_failed', 'qwen_failed', 'lmstudio_failed') OR rd.retry_count >= ?)
        )
        AND (f.mime_type LIKE 'image/%' OR LOWER(f.ext) IN ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif'))
        LIMIT ?
    """, (MAX_RETRIES, limit))
    all_rows.extend(c.fetchall())

    # 2. PDF
    if len(all_rows) < limit:
        c.execute("""
            SELECT r.file_id, f.path, r.reason
            FROM rag_ocr_status r JOIN files f ON r.file_id = f.id
            WHERE r.status = 'insufficient'
            AND NOT EXISTS (
                SELECT 1 FROM rag_descriptions rd 
                WHERE rd.file_id = r.file_id 
                AND (rd.source NOT IN ('vision_failed', 'qwen_failed', 'lmstudio_failed') OR rd.retry_count >= ?)
            )
            AND LOWER(f.ext) = '.pdf'
            LIMIT ?
        """, (MAX_RETRIES, limit - len(all_rows)))
        all_rows.extend(c.fetchall())

    # 3. Office
    if len(all_rows) < limit:
        placeholders = ','.join(['?'] * len(OFFICE_EXTS))
        c.execute(f"""
            SELECT r.file_id, f.path, r.reason
            FROM rag_ocr_status r JOIN files f ON r.file_id = f.id
            WHERE r.status = 'insufficient'
            AND NOT EXISTS (
                SELECT 1 FROM rag_descriptions rd 
                WHERE rd.file_id = r.file_id 
                AND (rd.source NOT IN ('vision_failed', 'qwen_failed', 'lmstudio_failed') OR rd.retry_count >= ?)
            )
            AND LOWER(f.ext) IN ({placeholders})
            LIMIT ?
        """, [MAX_RETRIES] + list(OFFICE_EXTS) + [limit - len(all_rows)])
        all_rows.extend(c.fetchall())

    # 4. djvu
    if len(all_rows) < limit:
        c.execute("""
            SELECT r.file_id, f.path, r.reason
            FROM rag_ocr_status r JOIN files f ON r.file_id = f.id
            WHERE r.status = 'insufficient'
            AND NOT EXISTS (
                SELECT 1 FROM rag_descriptions rd 
                WHERE rd.file_id = r.file_id 
                AND (rd.source NOT IN ('vision_failed', 'qwen_failed', 'lmstudio_failed') OR rd.retry_count >= ?)
            )
            AND LOWER(f.ext) = '.djvu'
            LIMIT ?
        """, (MAX_RETRIES, limit - len(all_rows)))
        all_rows.extend(c.fetchall())

    conn.close()
    return all_rows


# ============================================================
# 渲染函数
# ============================================================

def render_pdf_to_images(pdf_path: str) -> List[bytes]:
    try:
        import fitz
        with open(pdf_path, 'rb') as f:
            doc = fitz.open(stream=f.read(), filetype='pdf')
        total = len(doc)
        limit = min(total, MAX_DESCRIBE_PAGES)
        if total > MAX_DESCRIBE_PAGES:
            log.warning(f"PDF has {total} pages, clamping to {MAX_DESCRIBE_PAGES}: {pdf_path}")
        images = []
        for page_num in range(limit):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            images.append(pix.tobytes("png"))
        doc.close()
        log.info(f"Rendered {len(images)}/{total} pages from PDF: {pdf_path}")
        return images
    except Exception as e:
        log.warning(f"PDF render failed for {pdf_path}: {e}")
        return []


def render_office_to_images(office_path: str) -> List[bytes]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', tmpdir, office_path],
                capture_output=True, timeout=60
            )
            if result.returncode != 0:
                log.warning(f"LibreOffice conversion failed for {office_path}")
                return []
            pdf_files = list(Path(tmpdir).glob("*.pdf"))
            if not pdf_files:
                return []
            return render_pdf_to_images(str(pdf_files[0]))
    except Exception as e:
        log.warning(f"Office render failed for {office_path}: {e}")
        return []


def render_djvu_to_images(djvu_path: str) -> List[bytes]:
    images = []
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ['djvused', '-e', 'n', djvu_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return []
        try:
            total_pages = int(result.stdout.strip())
        except ValueError:
            return []
        for page_num in range(1, min(total_pages + 1, MAX_DESCRIBE_PAGES + 1)):
            out_path = os.path.join(tmpdir, f'page_{page_num}.png')
            result = subprocess.run(
                ['ddjvu', f'-page={page_num}', '-format=png', '-resolution=150',
                 djvu_path, out_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(out_path):
                with open(out_path, 'rb') as f:
                    images.append(f.read())
    return images


def render_file_to_images(file_path: str) -> List[bytes]:
    """根据文件类型渲染为图片列表。返回空列表表示不支持或失败。"""
    # 获取扩展名：优先用 suffix，对于 '.jpg' 这种特殊文件名（只有扩展名没有主文件名）
    # suffix 返回空字符串，需要用 name 作为 fallback
    p = Path(file_path)
    ext = p.suffix.lower()
    if not ext:
        # 文件名以 . 开头且没有其他 .，如 '.jpg'，把整个文件名当作扩展名
        name = p.name.lower()
        if name.startswith('.'):
            ext = name  # e.g. '.jpg'
        else:
            ext = ''

    if ext in IMAGE_EXTS:
        try:
            file_size = os.path.getsize(file_path)
            if file_size > MAX_IMAGE_FILE_BYTES:
                log.warning(f"Image too large ({file_size // 1024 // 1024}MB): {file_path}")
                return []
            with open(file_path, 'rb') as f:
                return [f.read()]
        except Exception as e:
            log.warning(f"Failed to read image {file_path}: {e}")
            return []
    elif ext == '.pdf':
        return render_pdf_to_images(file_path)
    elif ext in OFFICE_EXTS:
        return render_office_to_images(file_path)
    elif ext == '.djvu':
        return render_djvu_to_images(file_path)
    else:
        return []


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')


# ============================================================
# OpenAI 兼容 API 调用
# ============================================================

def call_openai_vision(endpoint: Dict[str, Any], image_b64: str, file_path: str,
                       page_num: int = 0, total_pages: int = 0) -> Optional[str]:
    """调用 OpenAI 兼容视觉模型
    
    Args:
        endpoint: 端点配置 dict
        image_b64: base64 编码的图片
        file_path: 文件路径
        page_num: 当前页码
        total_pages: 总页数
    
    Returns:
        生成的描述文本，或 None 表示失败
    """
    api_base = endpoint.get('api_base', '').rstrip('/')
    api_key = endpoint.get('api_key', '') or ''
    # 如果 api_key 为空，尝试从环境变量读取
    if not api_key:
        ep_name = endpoint.get('name', '')
        if 'qwen' in ep_name.lower() or 'dashscope' in api_base.lower():
            api_key = get_env('QWEN_API_KEY', '')
        elif 'lmstudio' in ep_name.lower() or 'localhost' in api_base:
            api_key = get_env('LMSTUDIO_API_KEY', '')
        else:
            api_key = get_env('VISION_API_KEY', '')
    model = endpoint.get('model', 'default')
    max_tokens = endpoint.get('max_tokens', 2048)
    timeout = endpoint.get('timeout', 60)
    name = endpoint.get('name', 'unknown')
    
    if not api_base:
        log.warning(f"[{name}] api_base not configured")
        return None
    
    # 构造请求
    page_info = f"（第 {page_num}/{total_pages} 页）" if total_pages > 1 else ""
    content = [
        {"type": "text", "text": f"请详细描述这个文件的内容{page_info}。文件: {os.path.basename(file_path)}。用中文描述，包括文字、图表、数据等。"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    url = f"{api_base}/chat/completions"
    
    for retry in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
                proxies={"http": None, "https": None} if api_base.startswith('http://localhost') else None
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                log.error(f"[{name}] Unexpected response: {result}")
                return None
            elif response.status_code == 429:
                # Rate limit
                if retry < MAX_RETRIES:
                    wait = min(60 * (2 ** retry), 300)  # 最大 5 分钟
                    log.warning(f"[{name}] 429 throttling, retry {retry+1}/{MAX_RETRIES} after {wait}s")
                    time.sleep(wait)
                    continue
                log.error(f"[{name}] 429 exhausted retries")
                return None
            else:
                log.error(f"[{name}] API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except requests.Timeout:
            log.warning(f"[{name}] Timeout ({timeout}s), retry {retry+1}/{MAX_RETRIES}")
            if retry < MAX_RETRIES:
                time.sleep(5)
                continue
            return None
        except Exception as e:
            log.error(f"[{name}] Call failed: {e}")
            if retry < MAX_RETRIES:
                time.sleep(5)
                continue
            return None
    
    return None


# ============================================================
# 端点选择与 fallback
# ============================================================

_endpoint_status = {}  # name -> {'available': bool, 'last_check': float}
_endpoint_lock = threading.Lock()
_last_call_time = {}  # name -> float (for rate limiting)


def check_endpoint_health(endpoint: Dict[str, Any]) -> bool:
    """检查端点是否可用"""
    api_base = endpoint.get('api_base', '').rstrip('/')
    name = endpoint.get('name', 'unknown')
    
    try:
        resp = requests.get(f"{api_base}/models", timeout=5, 
                           proxies={"http": None, "https": None} if api_base.startswith('http://localhost') else None)
        return resp.status_code == 200
    except Exception:
        return False


def select_endpoint() -> Optional[Dict[str, Any]]:
    """选择可用的端点（按顺序尝试）"""
    now = time.time()
    
    for endpoint in ENDPOINTS:
        name = endpoint.get('name', 'unknown')
        
        # 检查端点健康状态
        with _endpoint_lock:
            status = _endpoint_status.get(name, {})
            # 每 60 秒检查一次健康状态
            if not status or now - status.get('last_check', 0) > 60:
                available = check_endpoint_health(endpoint)
                _endpoint_status[name] = {'available': available, 'last_check': now}
            else:
                available = status.get('available', False)
        
        if not available:
            continue
        
        # 检查速率限制
        rate_limit = endpoint.get('rate_limit', 0)
        if rate_limit > 0:
            with _endpoint_lock:
                last_call = _last_call_time.get(name, 0)
                if now - last_call < rate_limit:
                    # 达到速率限制，跳过
                    continue
        
        return endpoint
    
    # 所有端点都不可用，返回第一个（最后尝试）
    return ENDPOINTS[0] if ENDPOINTS else None


def call_vision_with_fallback(image_b64: str, file_path: str,
                               page_num: int = 0, total_pages: int = 0) -> Tuple[Optional[str], Optional[str]]:
    """调用视觉模型（带 fallback）
    
    Returns:
        (description, endpoint_name) 或 (None, None)
    """
    # 尝试所有端点
    for endpoint in ENDPOINTS:
        name = endpoint.get('name', 'unknown')
        
        # 速率限制检查
        rate_limit = endpoint.get('rate_limit', 0)
        if rate_limit > 0:
            now = time.time()
            with _endpoint_lock:
                last_call = _last_call_time.get(name, 0)
                if now - last_call < rate_limit:
                    wait_time = rate_limit - (now - last_call)
                    log.info(f"[{name}] Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                _last_call_time[name] = time.time()
        
        result = call_openai_vision(endpoint, image_b64, file_path, page_num, total_pages)
        if result:
            return result, name
        else:
            log.warning(f"[{name}] Failed for {os.path.basename(file_path)}, trying next endpoint")
    
    return None, None


# ============================================================
# 处理单个文件
# ============================================================

def _process_file(file_id: int, file_path: str, reason: str) -> Tuple[bool, str]:
    """处理单个文件
    
    Returns:
        (success, endpoint_name)
    """
    if not os.path.exists(file_path):
        log.warning(f"File not found: {file_path}")
        return False, 'file_missing'

    images = render_file_to_images(file_path)
    if not images:
        log.warning(f"No images generated for {file_path}")
        # 标记为失败，避免无限重试
        fail_source = f"{backend.replace('_api', '')}_failed" if backend == 'qwen_api' else f"{backend}_failed"
        try:
            with _db_lock:
                conn = _get_db()
                c = conn.cursor()
                c.execute("""
                    INSERT INTO rag_descriptions (file_id, description, source, embedded, retry_count)
                    VALUES (?, '', ?, 0, 1)
                    ON CONFLICT(file_id) DO UPDATE SET
                        source = ?,
                        retry_count = retry_count + 1,
                        created_at = CURRENT_TIMESTAMP
                """, (file_id, fail_source, fail_source))
                conn.commit()
                conn.close()
        except Exception as e:
            log.error(f"Failed to write failure marker for {file_path}: {e}")
        return False, 'render_failed'

    images_b64 = [encode_image_to_base64(img) for img in images]
    del images
    total_images = len(images_b64)
    log.info(f"Describing {file_path}: {total_images} images")

    descriptions = []
    endpoint_used = None
    
    for i, img_b64 in enumerate(images_b64):
        page_num = i + 1
        desc, ep_name = call_vision_with_fallback(img_b64, file_path, page_num, total_images)
        if desc:
            if total_images > 1:
                descriptions.append(f"[Page {page_num}]\n{desc}")
            else:
                descriptions.append(desc)
            endpoint_used = ep_name
        else:
            log.warning(f"Failed to describe page {page_num}/{total_images}")

    full_description = "\n\n".join(descriptions)

    # 有效性检查
    if not full_description or len(full_description.strip()) < 20:
        log.warning(f"Invalid/short description for {file_path}: {len(full_description) if full_description else 0} chars")
        # 写入失败标记
        try:
            with _db_lock:
                conn = _get_db()
                c = conn.cursor()
                c.execute("""
                    INSERT INTO rag_descriptions (file_id, description, source, embedded, retry_count)
                    VALUES (?, ?, ?, 0, 1)
                    ON CONFLICT(file_id) DO UPDATE SET
                        description = excluded.description,
                        source = ?,
                        retry_count = retry_count + 1,
                        created_at = CURRENT_TIMESTAMP
                """, (file_id, full_description or '', 'vision_failed', 'vision_failed'))
                conn.commit()
                conn.close()
        except Exception as e:
            log.error(f"Failed to write failure marker for {file_path}: {e}")
        return False, 'vision_failed'

    # 写入成功
    try:
        with _db_lock:
            conn = _get_db()
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO rag_descriptions 
                (file_id, description, source, embedded)
                VALUES (?, ?, ?, 0)
            """, (file_id, full_description, endpoint_used or 'vision'))
            conn.commit()
            conn.close()
        log.info(f"✅ Described {file_path}: {len(full_description)} chars ({total_images} images) [{endpoint_used}]")
        del images_b64, descriptions, full_description
        gc.collect()
        return True, endpoint_used or 'vision'
    except Exception as e:
        log.error(f"DB write failed for {file_path}: {e}")
        return False, 'db_error'


# ============================================================
# 后台线程入口
# ============================================================

class DescriberThread(threading.Thread):
    """后台线程：持续处理 OCR 失败的文件
    
    使用多个端点 fallback，并行处理。
    """

    def __init__(self, name: str = "describer"):
        super().__init__(name=name, daemon=True)
        self._stop_event = threading.Event()
        self.processed = 0
        self.failed = 0
        self._endpoint_index = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info(f"🚀 Describer thread started ({len(ENDPOINTS)} endpoints)")

        while not self._stop_event.is_set():
            # 拉取待处理文件
            files = get_ocr_failed_files(limit=FETCH_LIMIT)
            if not files:
                log.info("📭 No more files to process, waiting 30s...")
                if self._stop_event.wait(timeout=30):
                    break
                continue

            log.info(f"📥 Found {len(files)} files to process")
            
            # 并行处理
            max_workers = MAX_WORKERS
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_file, fid, path, reason): (fid, path)
                    for fid, path, reason in files
                }
                for future in as_completed(futures):
                    if self._stop_event.is_set():
                        break
                    fid, path = futures[future]
                    try:
                        success, ep_name = future.result()
                        if success:
                            self.processed += 1
                        else:
                            self.failed += 1
                    except Exception as e:
                        log.error(f"Exception on {path}: {e}")
                        self.failed += 1

            log.info(f"📊 Progress: {self.processed} processed, {self.failed} failed")

            # 每 10 个文件做一次内存回收
            if (self.processed + self.failed) % 10 == 0:
                gc.collect()

            if self._stop_event.wait(timeout=2):
                break

        log.info(f"🛑 Describer thread stopped: {self.processed} processed, {self.failed} failed")


def start_describers() -> List[DescriberThread]:
    """启动 describer 后台线程。返回线程列表。
    
    单线程版本（使用多端点 fallback + 内部并行）。
    """
    threads = [DescriberThread(name="describer-main")]
    for t in threads:
        t.start()
    log.info(f"🚀 Describer threads started ({len(ENDPOINTS)} endpoints available)")
    return threads


def stop_describers(threads: List[DescriberThread], timeout: float = 120):
    """停止所有 describer 线程并等待退出。"""
    for t in threads:
        t.stop()
    for t in threads:
        t.join(timeout=timeout)
        if t.is_alive():
            log.warning(f"⚠️ {t.name} did not stop within {timeout}s")
        else:
            log.info(f"✅ {t.name} stopped: {t.processed} processed, {t.failed} failed")


# ============================================================
# 独立运行
# ============================================================

def main():
    """独立运行入口"""
    log.info("=" * 60)
    log.info("🚀 视觉描述生成器 (OpenAI 兼容 API)")
    log.info(f"端点: {len(ENDPOINTS)}")
    for ep in ENDPOINTS:
        log.info(f"  - {ep.get('name', 'unknown')}: {ep.get('api_base', '')} [{ep.get('model', 'default')}]")
    log.info("=" * 60)
    
    threads = start_describers()
    
    try:
        # 等待用户中断
        while True:
            time.sleep(60)
            log.info(f"📊 Running: {threads[0].processed} processed, {threads[0].failed} failed")
    except KeyboardInterrupt:
        log.info("Received interrupt signal, stopping...")
        stop_describers(threads, timeout=60)


if __name__ == "__main__":
    main()