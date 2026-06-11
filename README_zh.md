<div align="center">

# 🗂️ Local Multimodal RAG

**离线多模态文档检索——图片、PDF、Office 文档、代码。完全本地运行，无需云服务。**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()

[![English](https://img.shields.io/badge/English-DBEDFA)](./README.md) [![简体中文](https://img.shields.io/badge/简体中文-DFE0E5)](./README_zh.md)

</div>

---

<details open>
<summary><b>📕 目录</b></summary>

- [💡 这是什么？](#-这是什么)
- [✨ 核心特性](#-核心特性)
- [📸 工作原理](#-工作原理)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置说明](#-配置说明)
- [🌐 Web 界面与 API](#-web-界面与-api)
- [📊 对比](#-对比)
- [📁 项目结构](#-项目结构)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

</details>

---

## 💡 这是什么？

一套**完全离线的多模态检索增强生成（RAG）流水线**，能够处理图片、扫描版 PDF、Office 文档和代码——全部在你的机器上运行，无需云 API（云端模型可选用作回退）。

大多数 RAG 框架只能处理纯文本。本流水线填补了这一空白：

- 🖼️ **图片** → PaddleOCR 提取文字 → OCR 失败时回退到视觉模型描述
- 📄 **扫描版 PDF** → 渲染页面 → OCR → 文本提取
- 📊 **Office 文件**（docx/xlsx/pptx）→ LibreOffice 渲染 → OCR/视觉描述
- 💻 **代码与配置文件** → 语言感知的语义分块
- 🎯 **搜索** → 纯 NumPy 余弦相似度（零外部向量数据库依赖）

基于真实的线上生产经验构建——开箱即用即能处理 GPU 驱动崩溃、PaddleOCR 内存泄漏和子进程死锁。

## ✨ 核心特性

- 🔒 **100% 本地运行**——你的文件永不离开你的机器
- 🖼️ **多模态内容摄入**——图片、PDF、Office、代码、文本一体化流水线
- 🧠 **智能 OCR 回退**——PaddleOCR → 视觉模型描述 → 绝不丢失内容
- 🔍 **零依赖搜索**——纯 NumPy，无需 FAISS/Milvus/Chroma
- 🖥️ **Web 界面**——Google 风格搜索页面 + 配置面板
- 🚀 **REST API**——简单的搜索接口，方便集成
- 📡 **兼容 OpenAI 的视觉模型**——LMStudio / Ollama / vLLM / 任意 `/v1/chat/completions` 端点
- 🔄 **跨平台**——Linux 与 Windows（Windows 自动检测 Everything，回退到 os.walk）
- 📊 **基于 SQLite**——单一 `.db` 文件，备份和迁移轻松
- 🛡️ **久经考验**——内存泄漏隔离、GPU 崩溃防御、信号处理、恢复支持

## 📸 工作原理

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

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/yourname/local-multimodal-rag.git
cd local-multimodal-rag
pip install -r requirements.txt

# 可选：安装 PaddleOCR 用于图片文字提取
pip install paddlepaddle paddleocr

# 可选：安装 LibreOffice 用于 Office 文件渲染（Linux）
sudo apt install libreoffice
```

### 2. 配置

```bash
# 复制并编辑配置
cp .env.example .env
# 编辑 .env：添加 API 密钥（可选，仅用于云端视觉模型回退）

# 编辑 config.yaml：设置 scan_paths
# 默认扫描 ~/documents 和 ~/code
```

### 3. 运行

```bash
# 第 1 步：扫描文件到数据库
python run_pipeline.py scan

# 第 2 步：提取元数据
python run_pipeline.py extract-meta

# 第 3 步：处理与嵌入（主 RAG 流水线）
python run_pipeline.py process

# 第 4 步：通过 CLI 搜索
python run_pipeline.py search "你的关键词"

# 或者：启动 Web 服务器（http://localhost:8100）
python run_pipeline.py serve
```

**大功告成。** 在浏览器中打开 `http://localhost:8100`。

## ⚙️ 配置说明

所有设置均位于 `config.yaml`。关键配置段：

```yaml
# 扫描文件的位置
scanner:
  db_path: ./data/file_index.db
  scan_paths:
    - ~/documents
    - ~/code

# 视觉模型（任意兼容 OpenAI 的端点）
vision:
  endpoints:
    - api_base: "http://localhost:1234/v1"  # LMStudio / Ollama
      model: "qwen2.5-vl-7b"
      max_tokens: 1024

# Web 服务器
server:
  host: "0.0.0.0"
  port: 8100

# 嵌入
embedding:
  model_path: ./models/jina-embeddings-v5-text-small
  dim: 512
```

完整配置清单及注释请参见 [`config.yaml`](config.yaml)。

## 🌐 Web 界面与 API

### Web 界面

- **搜索页面**（`/`）——Google 风格搜索，支持文件类型筛选
- **配置面板**（`/config.html`）——浏览器中编辑所有设置

### REST API

```bash
# 搜索
curl "http://localhost:8100/api/search?q=machine+learning&top_k=50"

# 响应：
# {
#   "results": [
#     {"path": "/docs/paper.pdf", "score": 0.95, "snippet": "...", "file_type": "pdf"},
#     ...
#   ],
#   "meta": {"query": "machine learning", "total": 50, "elapsed_ms": 120}
# }

# 获取配置
curl http://localhost:8100/api/config

# 触发流水线
curl -X POST http://localhost:8100/api/pipeline/scan
curl -X POST http://localhost:8100/api/pipeline/process
```

## 📊 对比

| 特性 | **Local Multimodal RAG** | LangChain RAG | LlamaIndex | RAGFlow |
|---------|:------------------------:|:-------------:|:----------:|:-------:|
| 完全离线 | ✅ | ❌ | ❌ | ❌ |
| 图片 OCR → 文字 | ✅ | 插件 | 插件 | ✅ |
| 扫描版 PDF OCR | ✅ | 插件 | 插件 | ✅ |
| Office 文件 | ✅ | ❌ | 插件 | ✅ |
| 需要外部向量数据库 | ❌ | ✅ | ✅ | ✅ |
| GPU 崩溃防御 | ✅ | ❌ | ❌ | ❌ |
| 内存泄漏隔离 | ✅ | ❌ | ❌ | ❌ |
| 零配置搜索 | ✅ | ❌ | ❌ | ❌ |
| 内置 Web 界面 | ✅ | ❌ | ❌ | ✅ |
| 视觉模型回退 | ✅ | ❌ | ❌ | ✅ |
| Windows + Linux | ✅ | ✅ | ✅ | Docker |

## 📁 项目结构

```
local-multimodal-rag/
├── run_pipeline.py          # CLI 入口（scan/extract/process/search/serve）
├── server.py                # FastAPI Web 服务器
├── config.yaml              # 全部配置
├── config_loader.py         # 配置加载器 + 环境变量
│
├── file_scanner.py          # 文件发现（Everything / os.walk）
├── metadata_extractor.py    # EXIF、PDF 信息、媒体元数据
├── extractors.py            # 多模态内容提取
├── chunker_v2.py            # 语义分块
├── jina_v5_embedding.py     # Jina v5 文本嵌入
├── search_numpy.py          # NumPy 余弦搜索
├── describer.py             # 视觉模型（兼容 OpenAI）描述
├── description_embedder.py  # 描述文本嵌入
├── ocr_server.py            # PaddleOCR HTTP 服务
├── ocr_worker.py            # OCR 工作进程
│
├── static/                  # Web 界面
│   ├── index.html           # 搜索页面
│   ├── config.html          # 配置面板
│   └── style.css
│
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/amazing-thing`）
3. 提交你的修改
4. 发起 Pull Request

## 📄 许可证

[MIT License](LICENSE) —— 自由使用，随意发挥。
