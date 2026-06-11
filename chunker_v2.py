#!/usr/bin/env python3
"""
Greedy Semantic Chunker v2
1. Identify semantic boundaries (by file type)
2. Greedily merge adjacent units up to MAX_CHARS
3. Each chunk is as large as possible while keeping semantic integrity

Configuration loaded from config.yaml via config_loader.
"""

import re
from typing import List, Tuple

# --- Config ---
# Configuration loaded from config.yaml
try:
    from config_loader import get_chunking_config
    chunking_config = get_chunking_config()
    MAX_CHARS = chunking_config['max_chars']       # Greedy: merge up to configured chars
    OVERLAP_RATIO = chunking_config['overlap_ratio']  # Overlap ratio (e.g. 0.05 = 5%)
    OVERLAP_CHARS = int(MAX_CHARS * OVERLAP_RATIO)
except ImportError:
    # Fallback for standalone usage or testing
    MAX_CHARS = 12000     # Greedy: merge up to 12K chars
    OVERLAP_RATIO = 0.05  # 5% overlap = 600 chars at MAX_CHARS=12000
    OVERLAP_CHARS = int(MAX_CHARS * OVERLAP_RATIO)  # ~600 chars


def chunk_greedy_semantic(text: str, ext: str = '', max_chars: int = None) -> List[str]:
    """
    Greedy semantic chunking: merge semantic units up to MAX_CHARS.
    
    Pipeline:
    1. Split into semantic units (by file type)
    2. Greedily merge: accumulate units until next unit would exceed MAX_CHARS
    3. Result: each chunk = complete semantic units, as large as possible
    4. Add overlap: each chunk gets OVERLAP_CHARS from prev/next chunk (if exists)
    
    Args:
        text: Text to chunk
        ext: File extension (for semantic splitting logic)
        max_chars: Override MAX_CHARS (optional)
    """
    if not text or not text.strip():
        return []

    # Use override max_chars if provided
    effective_max_chars = max_chars if max_chars is not None else MAX_CHARS
    effective_overlap_chars = int(effective_max_chars * OVERLAP_RATIO)

    stripped = text.strip()

    # Entire text fits in one chunk (always emit, even tiny — path prefix has semantic value)
    if len(stripped) <= effective_max_chars:
        return [stripped]
    
    ext = ext.lower()
    
    # Step 1: Split into semantic units based on file type
    if ext in ('.md', '.rst', '.markdown'):
        units = _split_markdown_units(text)
    elif ext in ('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp',
                 '.cs', '.go', '.rs', '.rb', '.php', '.lua', '.sh', '.sql',
                 '.scala', '.kt', '.swift', '.dart'):
        units = _split_code_units(text, ext)
    else:
        units = _split_paragraph_units(text, max_chars=effective_max_chars)
    
    if not units:
        return [text.strip()]
    
    # Step 2: Greedy merge
    chunks = _greedy_merge(units, max_chars=effective_max_chars)
    
    # Step 3: Add overlap (prev/next chunk context)
    return _add_overlap(chunks, overlap_chars=effective_overlap_chars)


def _greedy_merge(units: List[Tuple[str, str]], max_chars: int = None) -> List[str]:
    """
    Greedy merge: accumulate units until adding the next would exceed MAX_CHARS.
    Safety: oversized single units are split by lines to enforce MAX_CHARS ceiling.
    
    units: list of (separator, content) tuples
    - separator: what goes between this unit and the previous (e.g. "\n\n")
    - content: the actual text
    
    Args:
        units: List of (separator, content) tuples
        max_chars: Override MAX_CHARS (optional)
    """
    effective_max_chars = max_chars if max_chars is not None else MAX_CHARS
    chunks = []
    current = ""
    
    for i, (sep, content) in enumerate(units):
        # Safety: if a single unit exceeds MAX_CHARS, split it by lines
        if len(content) > effective_max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            lines = content.split('\n')
            buf = ""
            for line in lines:
                if len(buf) + len(line) + 1 <= effective_max_chars:
                    buf = (buf + '\n' + line) if buf else line
                else:
                    if buf:
                        chunks.append(buf.strip())
                    buf = line
            if buf:
                chunks.append(buf.strip())
            continue
        
        # How much would adding this unit cost?
        addition = (sep + content) if current else content
        
        if not current:
            # Start new chunk
            current = content
        elif len(current) + len(addition) <= effective_max_chars:
            # Greedy: merge! Still fits.
            current = current + addition
        else:
            # Would exceed MAX_CHARS. Flush current, start new.
            chunks.append(current.strip())
            current = content
    
    if current:
        chunks.append(current.strip())
    
    # Filter empty chunks, but keep small ones (single chunk with path context)
    if chunks:
        return [c for c in chunks if c.strip()]
    elif units:
        return [units[0][1].strip()]
    else:
        return []


def _add_overlap(chunks: List[str], overlap_chars: int = None) -> List[str]:
    """
    Add overlap to each chunk: OVERLAP_CHARS from prev/next chunk (if exists).
    
    - Prepend last OVERLAP_CHARS of previous chunk (if has prev)
    - Append first OVERLAP_CHARS of next chunk (if has next)
    - Overlap is marked with [...] to distinguish from main content
    
    Args:
        chunks: List of chunk texts
        overlap_chars: Override OVERLAP_CHARS (optional)
    """
    effective_overlap_chars = overlap_chars if overlap_chars is not None else OVERLAP_CHARS
    
    if len(chunks) <= 1 or effective_overlap_chars == 0:
        return chunks
    
    overlapped = []
    for i, chunk in enumerate(chunks):
        # Build overlapped chunk
        parts = []
        
        # Prepend overlap from previous chunk
        if i > 0:
            prev_chunk = chunks[i - 1]
            prev_overlap = prev_chunk[-effective_overlap_chars:] if len(prev_chunk) > effective_overlap_chars else prev_chunk
            parts.append(f"[...{prev_overlap}]")
        
        # Main content
        parts.append(chunk)
        
        # Append overlap from next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            next_overlap = next_chunk[:effective_overlap_chars] if len(next_chunk) > effective_overlap_chars else next_chunk
            parts.append(f"[{next_overlap}...]")
        
        overlapped.append("\n".join(parts))
    
    return overlapped


# ============================================================
# Semantic Unit Splitters (by file type)
# ============================================================

def _split_markdown_units(text: str) -> List[Tuple[str, str]]:
    """Split markdown by headers + tables. Tables are extracted as atomic units
    to prevent sentence-level splitting from destroying table structure."""
    header_re = re.compile(r'^(#{1,4})\s+.+$', re.MULTILINE)
    # Markdown table: at least 2 lines with | delimiters + separator line
    table_re = re.compile(
        r'^(\|[^\n]+\|\n\|[-| :]+\|(?:\n\|[^\n]+\|)+)',
        re.MULTILINE
    )
    
    units = []
    last = 0
    current_heading = ""
    headers = list(header_re.finditer(text))
    
    for i, m in enumerate(headers):
        # If there's content before the first header (no heading context yet), emit it
        if m.start() > last and not current_heading:
            preamble = text[last:m.start()].strip()
            if preamble:
                units.append(("\n\n", preamble))
        # Section ends at the next header start, or end of text
        next_start = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_text = text[m.start():next_start]
        
        # Extract tables as separate atomic units within this section
        tables = list(table_re.finditer(section_text))
        if not tables:
            section = section_text.strip()
            if section:
                units.append(("\n\n", section))
        else:
            # Split section into non-table text + tables as separate units
            sec_offset = m.start()
            prev_end = 0
            for tm in tables:
                # Text before this table
                if tm.start() > prev_end:
                    pre_text = section_text[prev_end:tm.start()].strip()
                    if pre_text:
                        units.append(("\n\n", pre_text))
                # Table as atomic unit
                units.append(("\n\n", tm.group(1).strip()))
                prev_end = tm.end()
            # Text after last table
            if prev_end < len(section_text):
                post_text = section_text[prev_end:].strip()
                if post_text:
                    units.append(("\n\n", post_text))
        
        current_heading = m.group(0).strip()
        last = m.end()
    
    # If no headers found, fall back to paragraphs
    if not units:
        return _split_paragraph_units(text)
    
    return units


def _split_code_units(text: str, ext: str = '') -> List[Tuple[str, str]]:
    """Split code by top-level definitions. Uses per-language patterns."""
    
    # Language-specific patterns for top-level definitions
    if ext in ('.py',):
        # Python: decorators + def/class at top level (no leading whitespace)
        block_re = re.compile(
            r'^(?:(?:@\S+(?:\(.*?\))?\s*\n\s*)*(?:async\s+)?def\s+\w+'
            r'|(?:async\s+)?class\s+\w+)',
            re.MULTILINE
        )
    elif ext in ('.rs',):
        # Rust: fn (with or without pub), struct, enum, trait, mod, impl
        block_re = re.compile(
            r'^(?:(?:pub\s+)?(?:async\s+)?fn\s+\w+'
            r'|pub\s+(?:struct|enum|trait|mod|type)\s+\w+'
            r'|(?:pub\s+)?impl\s*(?:<[^>]*>\s*)?\S+'
            r'|mod\s+\w+)',
            re.MULTILINE
        )
    elif ext in ('.js', '.ts'):
        # JS/TS: function, class, export — but NOT plain var/let/const
        block_re = re.compile(
            r'^(?:(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+\w+'
            r'|(?:export\s+(?:default\s+)?)?class\s+\w+'
            r'|export\s+(?:const|let|var|function|class|interface|type|enum)\s+'
            r'|interface\s+\w+\s*(?:extends|\{)'
            r'|type\s+\w+\s*=)',
            re.MULTILINE
        )
    elif ext in ('.java', '.kt', '.scala', '.cs'):
        # JVM/C#: class, interface, enum, public methods
        block_re = re.compile(
            r'^(?:(?:public|private|protected|internal)?\s*(?:static\s+)?'
            r'(?:abstract\s+)?(?:final\s+)?(?:class|interface|enum|struct)\s+\w+'
            r'|(?:public|private|protected)?\s*(?:static\s+)?'
            r'(?:async\s+)?(?:\w+(?:<[^>]*>)?)\s+\w+\s*\()',
            re.MULTILINE
        )
    elif ext in ('.go',):
        # Go: func, type, var (at top level)
        block_re = re.compile(
            r'^(?:func\s+(?:\([^)]*\)\s+)?\w+'
            r'|type\s+\w+\s+(?:struct|interface)'
            r'|var\s+\w+\s+)',
            re.MULTILINE
        )
    elif ext in ('.c', '.cpp', '.h', '.hpp'):
        # C/C++: functions, classes, structs, namespaces
        block_re = re.compile(
            r'^(?:(?:static\s+|inline\s+)*(?:void|int|char|float|double|bool|auto|size_t)\s+\w+\s*\('
            r'|(?:class|struct|enum|namespace|union)\s+\w+'
            r'|(?:static\s+|inline\s+)*\w+\s*\*?\s*\w+\s*\()',  # pointer returns
            re.MULTILINE
        )
    else:
        # Generic: def, function, class only (skip var/let/const entirely)
        block_re = re.compile(
            r'^(?:(?:async\s+)?(?:def|function|class|interface|enum)\s+\w+)',
            re.MULTILINE
        )
    
    units = []
    splits = list(block_re.finditer(text))
    
    if not splits:
        # No clear structure → split by blank lines
        return _split_paragraph_units(text)
    
    last = 0
    for m in splits:
        if m.start() > last:
            preamble = text[last:m.start()].strip()
            if preamble:
                units.append(("\n\n", preamble))
        last = m.start()
    
    # The last block goes to the end
    if last < len(text):
        units.append(("\n\n", text[last:].strip()))
    
    return units if units else _split_paragraph_units(text)


def _split_paragraph_units(text: str, max_chars: int = None) -> List[Tuple[str, str]]:
    """Split by paragraphs (double newline). Fallback for all types."""
    effective_max_chars = max_chars if max_chars is not None else MAX_CHARS
    
    parts = re.split(r'(\n\s*\n)', text)
    units = []
    for part in parts:
        part = part.strip()
        if not part or part == '\n':
            continue
        units.append(("\n\n", part))

    # If a unit exceeds MAX_CHARS, split by sentences
    final = []
    for sep, content in units:
        if len(content) > effective_max_chars:
            sub_units = _split_sentence_units(content, max_chars=effective_max_chars)
            final.extend(sub_units)
        else:
            final.append((sep, content))

    return final if final else [("", text.strip())]


def _split_sentence_units(text: str, max_chars: int = None) -> List[Tuple[str, str]]:
    """Split very long text by sentence groups.
    Preserves markdown table structure — tables are not split by sentences."""
    effective_max_chars = max_chars if max_chars is not None else MAX_CHARS
    
    # Extract tables first
    table_re = re.compile(
        r'(\|[^\n]+\|\n\|[-| :]+\|(?:\n\|[^\n]+\|)+)',
        re.MULTILINE
    )
    tables = list(table_re.finditer(text))
    
    if not tables:
        # No tables — simple sentence split
        return _split_by_sentences(text, max_chars=effective_max_chars)
    
    # Text contains tables: split non-table parts by sentences, keep tables intact
    units = []
    prev_end = 0
    for tm in tables:
        # Split text before table by sentences
        if tm.start() > prev_end:
            before = text[prev_end:tm.start()]
            units.extend(_split_by_sentences(before, max_chars=effective_max_chars))
        # Table as atomic unit
        units.append(("\n\n", tm.group(1).strip()))
        prev_end = tm.end()
    # Text after last table
    if prev_end < len(text):
        units.extend(_split_by_sentences(text[prev_end:], max_chars=effective_max_chars))
    
    return units if units else _split_by_sentences(text, max_chars=effective_max_chars)


def _split_by_sentences(text: str, max_chars: int = None) -> List[Tuple[str, str]]:
    """Split text into sentence-level units for greedy merging."""
    effective_max_chars = max_chars if max_chars is not None else MAX_CHARS
    
    sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
    units = []
    current = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(current) + len(s) + 1 > effective_max_chars and len(current) > 0:
            units.append((" ", current))
            current = s
        else:
            current = (current + " " + s) if current else s
    if current:
        units.append((" ", current))
    return units


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    import sqlite3, os, sys
    
    # Test with real files from DB
    db_path = sys.argv[1] if len(sys.argv) > 1 else './file_index.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    for label, ext in [('Markdown', '.md'), ('Python', '.py'), ('Text', '.txt')]:
        c.execute(f"SELECT path, name, size FROM files WHERE is_deleted=0 AND LOWER(ext)='{ext}' AND size > 5000 AND size < 500000 ORDER BY RANDOM() LIMIT 5")
        
        print(f"\n=== {label} ===")
        for path, name, size in c.fetchall():
            if not os.path.exists(path):
                continue
            for enc in ['utf-8', 'gb18030', 'latin-1']:
                try:
                    with open(path, 'r', encoding=enc, errors='replace') as f:
                        text = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            chunks = chunk_greedy_semantic(text, ext=ext)
            sizes = [len(c) for c in chunks]
            print(f"  {name[:35]:35} {size:>8}B → {len(chunks)} chunks, sizes: {sizes}")
    
    conn.close()