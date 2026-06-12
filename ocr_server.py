#!/usr/bin/env python3
"""
PaddleOCR GPU API Server — persistent service for RAG pipeline.

Uses multiprocessing.Pool with spawn mode to enable parallel OCR on GPU.
Each worker process independently initializes CUDA and loads PaddleOCR.

Usage:
    python3 ocr_server.py [--port 8002] [--workers 4]

API:
    POST /ocr
    Body: {"images": ["/path/to/img1.jpg", "/path/to/img2.png", ...]}
    Response: {"results": ["text1", null, "text3", ...]}  (null = failed/no text)

    GET /health  → 200 OK
"""

import os
import sys
import json
import argparse
import logging
import multiprocessing
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger("ocr_server")

# Global worker pool
_worker_pool = None
_worker_count = 4  # default, will be set in main()


def _restart_worker_pool():
    """Restart the worker pool when workers get stuck."""
    global _worker_pool, _worker_count
    log.info("Restarting worker pool...")
    
    # Terminate the old pool (force kill workers)
    if _worker_pool is not None:
        try:
            _worker_pool.terminate()  # Send SIGTERM to workers
            _worker_pool.join()       # Wait for them to exit (no timeout param)
        except Exception as e:
            log.warning(f"Error terminating old pool: {e}")
    
    # Create new pool
    ctx = multiprocessing.get_context('spawn')
    _worker_pool = ctx.Pool(processes=_worker_count, initializer=init_worker)
    log.info(f"Worker pool restarted with {_worker_count} processes")


def init_worker():
    """Initialize PaddleOCR in each worker process."""
    global _ocr
    log.info(f"Worker {os.getpid()}: Loading PaddleOCR (GPU mode)...")
    from paddleocr import PaddleOCR
    # 抑制 PaddleOCR 内部的 angle classifier WARNING（每张图片都输出一次）
    import logging as _logging
    _logging.getLogger('ppocr').setLevel(_logging.ERROR)
    _ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='ch',
        use_gpu=True,
        cls=True,
    )
    log.info(f"Worker {os.getpid()}: ✅ PaddleOCR loaded (GPU mode)")


def ocr_single_image(image_path: str) -> str | None:
    """OCR a single image in worker process."""
    global _ocr
    if not os.path.isfile(image_path):
        return None
    try:
        result = _ocr.ocr(image_path, det=True, rec=True, cls=True)
        if result and result[0]:
            texts = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    if text.strip():
                        texts.append(text.strip())
            return "\n".join(texts) if texts else None
        return None
    except Exception as e:
        log.warning(f"OCR failed for {image_path}: {e}")
        return None


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class OCRHandler(BaseHTTPRequestHandler):
    """HTTP handler for OCR API."""

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_POST(self):
        if self.path == '/ocr':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)
                image_paths = data.get('images', [])

                if not image_paths:
                    self._json_response({"results": []})
                    return

                # Use map_async with timeout to avoid infinite blocking
                # If a worker gets stuck, we need to restart the pool
                TIMEOUT_PER_IMAGE = 30  # seconds per image
                timeout = max(60, len(image_paths) * TIMEOUT_PER_IMAGE)
                
                try:
                    async_result = _worker_pool.map_async(ocr_single_image, image_paths)
                    results = async_result.get(timeout=timeout)
                    self._json_response({"results": results})
                except multiprocessing.TimeoutError:
                    log.error(f"OCR timeout after {timeout}s for {len(image_paths)} images, restarting worker pool...")
                    # Restart the worker pool to recover from stuck workers
                    _restart_worker_pool()
                    self._json_response({"results": [None] * len(image_paths), "error": "timeout"})
                except Exception as pool_err:
                    log.error(f"Pool error: {pool_err}, restarting worker pool...")
                    _restart_worker_pool()
                    self._json_response({"results": [None] * len(image_paths), "error": str(pool_err)})

            except Exception as e:
                log.error(f"Request error: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)

    def _json_response(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        # Suppress default HTTP request logging
        pass


def main():
    global _worker_pool, _worker_count
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8002, help='Server port (default: 8002)')
    parser.add_argument('--workers', type=int, default=4, help='Number of OCR workers (default: 4)')
    args = parser.parse_args()

    _worker_count = args.workers
    log.info(f"🚀 Starting OCR API server on port {args.port} with {args.workers} workers (spawn mode)...")
    
    # Use spawn context to avoid CUDA context loss from fork
    ctx = multiprocessing.get_context('spawn')
    
    # Create worker pool (each worker loads its own PaddleOCR instance)
    _worker_pool = ctx.Pool(processes=args.workers, initializer=init_worker)
    log.info(f"✅ Worker pool created with {args.workers} processes (spawn mode)")

    server = ThreadingHTTPServer(('127.0.0.1', args.port), OCRHandler)
    server.socket.listen(128)
    log.info(f"✅ OCR API server ready on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        _worker_pool.close()
        _worker_pool.join()
        server.server_close()


if __name__ == '__main__':
    main()
