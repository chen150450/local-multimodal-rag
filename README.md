<div align="center">

# 🗂️ Local Multimodal RAG

**离线多媒体文档检索 — 图片、PDF、Office、代码，全部本地处理，不碰云**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()

</div>

---

<details open>
<summary><b>📕 Table of Contents</b></summary>

- [💡 What is this?](#-what-is-this)
- [✨ Key Features](#-key-features)
- [📸 How it Works](#-how-it-works)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#-configuration)
- [🌐 Web UI & API](#-web-ui--api)
- [📊 Comparison](#-comparison)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

</details>

---

## 💡 What is this?

A **fully local, multimodal Retrieval-Augmented Generation (RAG) pipeline** that processes images, scanned PDFs, Office documents, and code — all on your machine, no cloud API required (cloud models optional as fallback).

Most RAG frameworks only handle plain text. This pipeline fills the gap:

- 🖼️ **Images** → PaddleOCR extracts text → OCR failures fall back to vision model descriptions
- 📄 **Scanned PDFs** → Render pages → OCR → text extraction
- 📊 **Office files** (docx/xlsx/pptx) → LibreOffice render → OCR/visual description
- 💻 **Code & configs** → Language-aware semantic chunking
- 🎯 **Search** → Pure NumPy cosine similarity (zero external vector DB dependency)

Built from real-world production experience — handles GPU driver crashes, PaddleOCR memory leaks, and subprocess deadlocks out of the box.

## ✨ Key Features

- 🔒 **100% local** — Your files never leave your machine
- 🖼️ **Multimodal ingestion** — Images, PDFs, Office, code, text in one pipeline
- 🧠 **Smart OCR fallback** — PaddleOCR → vision model description → never lose content
- 🔍 **Zero-dependency search** — Pure NumPy, no FAISS/Milvus/Chroma needed
- 🖥️ **Web UI** — Google-style search page + config dashboard
- 🚀 **REST API** — Simple search endpoint for integration
- 📡 **OpenAI-compatible vision** — LMStudio / Ollama / vLLM / any `/v1/chat/completions` endpoint
- 🔄 **Cross-platform** — Linux & Windows (auto-detects Everything on Windows, falls back to os.walk)
- 📊 **SQLite-based** — Single `.db` file, easy backup and transfer
- 🛡️ **Battle-tested** — Memory leak isolation, GPU crash defense, signal handling, resume support

## 📸 How it Works

```
┌─────────────────────────────────────────────────────────┐
│                    File Scanner                          │
│   Windows: Everything SDK  │  Linux: os.walk             │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│                 Metadata Extractor                       │
│         EXIF / PDF info / Media / Code stats             │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              Multimodal Extractor                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ PaddleOCR │  │ PyMuPDF  │  │LibreOffice│  │  Code  │  │
│  │ (images)  │  │  (PDF)   │  │ (Office)  │  │ Parser │  │
│  └─────┬─────┘  └────┬─────┘  └────┬──────┘  └───┬────┘  │
│        │              │              │              │       │
│        ▼              ▼              ▼              ▼       │
│   ┌──────────────────────────────────────────────────┐    │
│   │  OCR Quality Check                               │    │
│   │  Failed? → Vision Model (OpenAI-compat API)      │    │
│   └──────────────────────────────────────────────────┘    │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              Semantic Chunker                            │
│    Greedy merging · Code-aware · 1200 char max          │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│            Jina v5 Embedding (512-dim)                   │
│      Local model · Matryoshka · int8 quantized          │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│            SQLite + NumPy Search                         │
│     Cosine similarity · No external vector DB needed     │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/yourname/local-multimodal-rag.git
cd local-multimodal-rag
pip install -r requirements.txt

# Optional: Install PaddleOCR for image text extraction
pip install paddlepaddle paddleocr

# Optional: Install LibreOffice for Office file rendering (Linux)
sudo apt install libreoffice
```

### 2. Configure

```bash
# Copy and edit config
cp .env.example .env
# Edit .env: add API keys (optional, only for cloud vision fallback)

# Edit config.yaml: set your scan_paths
# Default: scans ~/documents and ~/code
```

### 3. Run

```bash
# Step 1: Scan files into database
python run_pipeline.py scan

# Step 2: Extract metadata
python run_pipeline.py extract-meta

# Step 3: Process & embed (the main RAG pipeline)
python run_pipeline.py process

# Step 4: Search via CLI
python run_pipeline.py search "your keywords"

# Or: Start web server (http://localhost:8100)
python run_pipeline.py serve
```

**That's it.** Open `http://localhost:8100` in your browser.

## ⚙️ Configuration

All settings live in `config.yaml`. Key sections:

```yaml
# Where to scan files
scanner:
  db_path: ./data/file_index.db
  scan_paths:
    - ~/documents
    - ~/code

# Vision model (any OpenAI-compatible endpoint)
vision:
  endpoints:
    - api_base: "http://localhost:1234/v1"  # LMStudio / Ollama
      model: "qwen2.5-vl-7b"
      max_tokens: 1024

# Web server
server:
  host: "0.0.0.0"
  port: 8100

# Embedding
embedding:
  model_path: ./models/jina-embeddings-v5-text-small
  dim: 512
```

See [`config.yaml`](config.yaml) for the full list with comments.

## 🌐 Web UI & API

### Web Interface

- **Search page** (`/`) — Google-style search with file type filters
- **Config dashboard** (`/config.html`) — Edit all settings in browser

### REST API

```bash
# Search
curl "http://localhost:8100/api/search?q=machine+learning&top_k=50"

# Response:
# {
#   "results": [
#     {"path": "/docs/paper.pdf", "score": 0.95, "snippet": "...", "file_type": "pdf"},
#     ...
#   ],
#   "meta": {"query": "machine learning", "total": 50, "elapsed_ms": 120}
# }

# Get config
curl http://localhost:8100/api/config

# Trigger pipeline
curl -X POST http://localhost:8100/api/pipeline/scan
curl -X POST http://localhost:8100/api/pipeline/process
```

## 📊 Comparison

| Feature | **Local Multimodal RAG** | LangChain RAG | LlamaIndex | RAGFlow |
|---------|:------------------------:|:-------------:|:----------:|:-------:|
| Fully offline | ✅ | ❌ | ❌ | ❌ |
| Image OCR → text | ✅ | Plugin | Plugin | ✅ |
| Scanned PDF OCR | ✅ | Plugin | Plugin | ✅ |
| Office files | ✅ | ❌ | Plugin | ✅ |
| External vector DB needed | ❌ | ✅ | ✅ | ✅ |
| GPU crash defense | ✅ | ❌ | ❌ | ❌ |
| Memory leak isolation | ✅ | ❌ | ❌ | ❌ |
| Zero-config search | ✅ | ❌ | ❌ | ❌ |
| Built-in web UI | ✅ | ❌ | ❌ | ✅ |
| Vision model fallback | ✅ | ❌ | ❌ | ✅ |
| Windows + Linux | ✅ | ✅ | ✅ | Docker |

## 📁 Project Structure

```
local-multimodal-rag/
├── run_pipeline.py          # CLI entry point (scan/extract/process/search/serve)
├── server.py                # FastAPI web server
├── config.yaml              # All configuration
├── config_loader.py         # Config loader + env vars
│
├── file_scanner.py          # File discovery (Everything / os.walk)
├── metadata_extractor.py    # EXIF, PDF info, media metadata
├── extractors.py            # Multimodal content extraction
├── chunker_v2.py            # Semantic chunking
├── jina_v5_embedding.py     # Jina v5 text embedding
├── search_numpy.py          # NumPy cosine search
├── describer.py             # Vision model (OpenAI-compat) descriptions
├── description_embedder.py  # Description text embedding
├── ocr_server.py            # PaddleOCR HTTP service
├── ocr_worker.py            # OCR worker process
│
├── static/                  # Web UI
│   ├── index.html           # Search page
│   ├── config.html          # Config dashboard
│   └── style.css
│
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## 🤝 Contributing

Issues and pull requests are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-thing`)
3. Commit your changes
4. Open a PR

## 📄 License

[MIT License](LICENSE) — use it however you want.
