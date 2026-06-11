#!/usr/bin/env python3
"""
Standalone OCR subprocess worker.

Runs PaddleOCR in a completely isolated process.
Called by extractors.py via subprocess.run().

Memory is freed completely when this process exits — no leak accumulation.

Configuration loaded from config.yaml via config_loader.

Usage:
    echo '<json>' | python3 ocr_worker.py
    python3 ocr_worker.py path1.jpg path2.png ...

Input:  JSON list of image paths on stdin, OR file paths as CLI args
Output: JSON list of results to stdout (each item is string or null)
"""

import os
import sys
import json

# Import config loader for OCR settings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_loader import get_config
    config = get_config()
    ocr_config = config['ocr']
    OCR_LANG = ocr_config['lang']
    OCR_USE_GPU = ocr_config.get('use_gpu', True)
except ImportError:
    OCR_LANG = 'ch'
    OCR_USE_GPU = True


# PaddleOCR GPU 模式：subprocess 隔离，退出后显存自动释放
def main():
    # Read image paths
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        # Read JSON from stdin
        try:
            data = json.load(sys.stdin)
            image_paths = data if isinstance(data, list) else [data]
        except Exception:
            print("[]", flush=True)
            sys.exit(1)

    if not image_paths:
        print("[]", flush=True)
        sys.exit(0)

    # Import PaddleOCR (GPU mode)
    # 重定向 PaddleOCR stdout → stderr，确保 stdout 只有我们的 JSON 输出
    import io
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr  # PaddleOCR 的日志输出到 stderr
    
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=OCR_LANG,
        use_gpu=OCR_USE_GPU,
        show_log=False,
    )
    
    sys.stdout = _real_stdout  # 恢复 stdout 给我们的 JSON 输出

    results = []
    for p in image_paths:
        if not os.path.isfile(p):
            results.append(None)
            continue
        try:
            page_result = ocr.ocr(p, det=True, rec=True, cls=True)
            if page_result and page_result[0]:
                texts = []
                for line in page_result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        if text.strip():
                            texts.append(text.strip())
                results.append("\n".join(texts) if texts else None)
            else:
                results.append(None)
        except Exception as e:
            results.append(None)

    # Output JSON — 只输出最后一行（跳过 PaddleOCR 的 WARNING/stdout 日志）
    json_line = json.dumps(results, ensure_ascii=False)
    # 用空行分隔，确保 JSON 是最后一行
    sys.stdout.write(json_line + '\n')
    sys.stdout.flush()


if __name__ == "__main__":
    main()