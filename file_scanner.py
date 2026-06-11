#!/usr/bin/env python3
"""
文件扫描器 — 跨平台版本
- Windows: 使用 Everything (es.exe) 高速索引（可选）
- Linux/Mac: 纯 os.walk 扫描
- 数据库同步：新增/删除/变更检测

调用:
  python file_scanner.py
  python file_scanner.py --dry-run
"""

import sqlite3
import os
import re
import sys
import subprocess
import time
import shutil
import stat
import gc
import platform
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 配置加载
try:
    from config_loader import get_config, get_env
    config = get_config()
    scanner_config = config.get('scanner', {})
    DB_PATH = config.get('database', {}).get('path', './file_index.db')
    SCAN_PATHS = scanner_config.get('scan_paths', [])
    ES_EXE = scanner_config.get('everything_exe', '')
    HOME_PATHS = scanner_config.get('home_paths', [])
    SKIP_DIRS = set(scanner_config.get('skip_dirs', ['.git', 'node_modules', '__pycache__', '.cache', '.npm']))
    BATCH_SIZE = 10000
except ImportError:
    # 独立运行时的默认配置
    DB_PATH = "./file_index.db"
    SCAN_PATHS = []
    ES_EXE = ""
    HOME_PATHS = []
    SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.cache', '.npm'}
    BATCH_SIZE = 10000

_PATH_RE = re.compile(r'^([A-Z]):[/\\](.*)$')
_FT_OFFSET = 116444736000000000  # FILETIME to Unix epoch offset (100ns intervals)

EXTRA_MIME = {
    '.md':'text/markdown','.json':'application/json','.yaml':'application/yaml',
    '.yml':'application/yaml','.toml':'application/toml','.log':'text/plain',
    '.cfg':'text/plain','.ini':'text/plain','.conf':'text/plain',
    '.sh':'text/x-shellscript','.py':'text/x-python','.js':'application/javascript',
    '.ts':'application/typescript','.tsx':'application/typescript',
    '.jsx':'application/javascript','.c':'text/x-c','.cpp':'text/x-c++',
    '.h':'text/x-c','.hpp':'text/x-c++','.cs':'text/x-csharp',
    '.java':'text/x-java','.kt':'text/x-kotlin','.rs':'text/x-rust',
    '.go':'text/x-go','.rb':'text/x-ruby','.php':'text/x-php',
    '.lua':'text/x-lua','.sql':'application/sql','.swift':'text/x-swift',
    '.dart':'application/dart','.vue':'text/x-vue','.svelte':'text/x-svelte',
    '.exe':'application/x-msdos-program','.dll':'application/x-msdownload',
    '.ttf':'font/ttf','.otf':'font/otf','.woff':'font/woff',
    '.woff2':'font/woff2','.torrent':'application/x-bittorrent',
    '.apk':'application/vnd.android.package-archive',
    '.7z':'application/x-7z-compressed','.rar':'application/x-rar',
    '.zip':'application/zip','.gz':'application/gzip',
    '.xz':'application/x-xz','.bz2':'application/x-bzip2',
    '.tar':'application/x-tar','.iso':'application/x-iso9660-image',
    '.wav':'audio/wav','.flac':'audio/flac','.mp3':'audio/mpeg',
    '.ogg':'audio/ogg','.aac':'audio/aac','.m4a':'audio/mp4','.opus':'audio/opus',
    '.mp4':'video/mp4','.mkv':'video/x-matroska','.avi':'video/x-msvideo',
    '.mov':'video/quicktime','.wmv':'video/x-ms-wmv','.webm':'video/webm',
    '.flv':'video/x-flv','.rmvb':'application/vnd.rn-realmedia-vbr',
    '.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png',
    '.gif':'image/gif','.webp':'image/webp','.bmp':'image/bmp',
    '.svg':'image/svg+xml','.ico':'image/x-icon','.tiff':'image/tiff',
    '.heic':'image/heic','.heif':'image/heif','.psd':'image/vnd.adobe.photoshop',
    '.doc':'application/msword',
    '.docx':'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls':'application/vnd.ms-excel',
    '.xlsx':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.ppt':'application/vnd.ms-powerpoint',
    '.pptx':'application/vnd.openxmlformats-officedocument.presentationml.sheet',
    '.pdf':'application/pdf',
    '.epub':'application/epub+zip','.mobi':'application/x-mobipocket-ebook',
    '.bat':'application/x-bat','.cmd':'application/x-bat','.ps1':'application/x-powershell',
    '.jar':'application/x-java-archive','.pem':'application/x-x509-ca-cert',
    '.crt':'application/x-x509-ca-cert','.nfo':'text/x-nfo',
    '.m3u':'application/x-mpegurl','.m3u8':'application/x-mpegurl',
    '.msi':'application/x-msi','.dmg':'application/x-apple-diskimage',
    '.deb':'application/vnd.debian-binary-package',
    '.rpm':'application/x-rpm','.chm':'application/x-chm',
    '.djvu':'image/vnd.djvu',
}

import mimetypes
for _e, _m in list(EXTRA_MIME.items()):
    mimetypes.add_type(_m, _e)


def convert_path(wp: str) -> str:
    """Windows 路径转 WSL 路径"""
    m = _PATH_RE.match(wp)
    if not m:
        return wp
    return f'/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), "/")}'


def get_mime(ext: str) -> str:
    """根据扩展名获取 MIME 类型"""
    if ext:
        el = ext.lower()
        if el in EXTRA_MIME:
            return EXTRA_MIME[el]
    if ext:
        try:
            mt, _ = mimetypes.guess_type('x' + ext)
            if mt:
                return mt
        except Exception:
            pass
    return 'application/octet-stream'


def ft_to_unix(ft_str: str) -> float:
    """FILETIME (100ns since 1601) → Unix timestamp"""
    if not ft_str or not ft_str.strip():
        return None
    try:
        ft = int(ft_str)
        if ft <= 0:
            return None
        return (ft - _FT_OFFSET) / 1e7
    except (ValueError, TypeError):
        return None


def has_everything() -> bool:
    """检查 Everything (es.exe) 是否可用"""
    if not ES_EXE:
        return False
    # Windows only
    if platform.system() != 'Windows':
        return False
    try:
        result = subprocess.run([ES_EXE, '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def export_drive_with_everything(drive: str, export_dir: str) -> tuple:
    """使用 Everything 导出单个盘到 CSV"""
    out = os.path.join(export_dir, f"{drive}.csv")
    # 跳过已有且足够大的 CSV（>1MB）
    if os.path.exists(out) and os.path.getsize(out) > 1048576:
        return drive, os.path.getsize(out), 0
    
    cmd = [
        ES_EXE,
        '-size', '-dm', '-dc', '-da',
        '-csv',
        '-no-digit-grouping',
        '-date-format', '2',  # FILETIME integer
        '-export-csv', out,
        '/a-d',  # files only
        f'{drive}:'
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    return drive, sz, proc.returncode


def scan_path_oswalk(scan_path: str) -> list:
    """纯 os.walk 扫描指定路径
    返回：[(path, name, ext, size, mtime, ctime, atime, depth, mime_type), ...]
    """
    files = []
    scan_path = os.path.expanduser(scan_path)
    if not os.path.exists(scan_path):
        return files
    
    for root, dirs, filenames in os.walk(scan_path, followlinks=False):
        # 跳过指定目录
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        depth = root.count(os.sep) - scan_path.count(os.sep)
        for fname in filenames:
            fpath = os.path.join(root, fname)
            try:
                s = os.lstat(fpath)
                if not stat.S_ISREG(s.st_mode):
                    continue
                dot_idx = fname.rfind('.')
                ext = fname[dot_idx:].lower() if dot_idx > 0 else None
                files.append((
                    fpath, fname, ext, s.st_size,
                    s.st_mtime, s.st_ctime, s.st_atime,
                    depth, get_mime(ext)
                ))
            except (OSError, IOError):
                continue
    return files


def parse_and_insert_csv(csv_path: str, cur, conn, batch_size=10000) -> tuple:
    """流式解析 CSV 并插入数据库
    返回：(valid_count, line_count)
    """
    valid_count = 0
    line_count = 0
    batch = []
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        # Skip header
        header = f.readline()
        if not header.startswith('Size'):
            f.seek(0)
        
        for line in f:
            line_count += 1
            try:
                parts = line.rstrip('\n\r').split(',', 4)
                if len(parts) < 5:
                    continue

                size_s, dm_s, dc_s, da_s, filename = parts

                # Parse size
                size = None
                if size_s.strip():
                    try:
                        size_val = int(size_s)
                        if 0 <= size_val <= 9223372036854775807:
                            size = size_val
                    except ValueError:
                        pass

                # Parse filename
                filename = filename.strip().strip('"')
                if not filename or not isinstance(filename, str):
                    continue

                # Convert path
                wsl = convert_path(filename)
                if not wsl or not isinstance(wsl, str):
                    continue
                try:
                    name = os.path.basename(wsl)
                    dot_idx = name.rfind('.')
                    ext = name[dot_idx:].lower() if dot_idx > 0 else None
                    depth = wsl.count('/') - 3
                except TypeError:
                    continue

                batch.append((
                    wsl, name, ext, size,
                    ft_to_unix(dm_s), ft_to_unix(dc_s), ft_to_unix(da_s),
                    depth, get_mime(ext)
                ))
                valid_count += 1
                
                if len(batch) >= batch_size:
                    cur.executemany(
                        "INSERT OR IGNORE INTO _ev_temp VALUES(?,?,?,?,?,?,?,?,?)",
                        batch
                    )
                    conn.commit()
                    batch.clear()
                    gc.collect()
                    
            except (ValueError, IndexError):
                continue
        
        if batch:
            cur.executemany(
                "INSERT OR IGNORE INTO _ev_temp VALUES(?,?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
            batch.clear()
    
    gc.collect()
    return valid_count, line_count


def sync(dry_run=False):
    """主同步函数"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{'='*60}", flush=True)
    print(f"文件扫描器 v1 (跨平台) | {ts}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Phase 0: 初始化 ──
    print("\n[0] 初始化数据库...", flush=True)
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=600000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA temp_store=FILE")
    cur = conn.cursor()

    # 创建 files 表（如果不存在）
    cur.execute("""CREATE TABLE IF NOT EXISTS files(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        name TEXT, ext TEXT, size INTEGER,
        mtime REAL, ctime REAL, atime REAL,
        is_dir INTEGER DEFAULT 0,
        depth INTEGER,
        scanned_at TEXT,
        is_deleted INTEGER DEFAULT 0,
        file_hash TEXT,
        mime_type TEXT,
        media_meta TEXT
    )""")
    conn.commit()

    # 清理临时表
    for tbl in ['_ev_temp', '_to_delete']:
        try:
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        except Exception:
            pass
    conn.commit()

    # 统计当前状态
    cur.execute("SELECT COUNT(*) FROM files")
    total_before = cur.fetchone()[0]
    print(f"  DB 当前：{total_before:,} 条", flush=True)

    if dry_run:
        print("DRY-RUN 模式，退出", flush=True)
        conn.close()
        return

    # ── Phase 1: 扫描所有路径 ──
    print("\n[1] 扫描所有路径...", flush=True)
    t0 = time.time()
    
    # 创建临时表
    cur.execute("""CREATE TABLE _ev_temp(
        path TEXT PRIMARY KEY,
        name TEXT, ext TEXT, size INTEGER,
        mtime REAL, ctime REAL, atime REAL,
        depth INTEGER, mime_type TEXT
    ) WITHOUT ROWID""")
    conn.commit()
    
    total_scanned = 0
    use_everything = has_everything()
    
    # Windows 盘列表（如果有 Everything）
    drives = []
    if use_everything and platform.system() == 'Windows':
        # 检测可用的盘符
        for letter in 'CDEFGH':
            drive_path = f"{letter}:\\"
            if os.path.exists(drive_path):
                drives.append(letter)
    
    # Everything 导出（仅 Windows）
    if use_everything and drives:
        export_dir = os.path.join(os.environ.get('TEMP', '/tmp'), 'everything_export')
        os.makedirs(export_dir, exist_ok=True)
        print(f"  使用 Everything 导出 {len(drives)} 个盘...", flush=True)
        
        with ProcessPoolExecutor(max_workers=min(6, len(drives))) as pool:
            futures = {pool.submit(export_drive_with_everything, d, export_dir): d for d in drives}
            for fut in as_completed(futures):
                drive, sz, rc = fut.result()
                print(f"  盘 {drive}: {sz/1048576:.1f} MB (rc={rc})", flush=True)
        
        # 解析 CSV
        for drive in drives:
            csv_path = os.path.join(export_dir, f"{drive}.csv")
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10:
                continue
            valid_count, line_count = parse_and_insert_csv(csv_path, cur, conn)
            total_scanned += valid_count
            print(f"  盘 {drive}: {valid_count:,} 文件", flush=True)
        
        shutil.rmtree(export_dir, ignore_errors=True)
    
    # os.walk 扫描所有配置的路径
    print(f"  使用 os.walk 扫描 {len(SCAN_PATHS) + len(HOME_PATHS)} 个路径...", flush=True)
    for scan_path in SCAN_PATHS + HOME_PATHS:
        scan_path = os.path.expanduser(scan_path)
        if not os.path.exists(scan_path):
            print(f"  跳过 {scan_path} (不存在)", flush=True)
            continue
        
        files = scan_path_oswalk(scan_path)
        if files:
            # 批量插入
            batch = []
            for f in files:
                batch.append(f)
                if len(batch) >= 50000:
                    cur.executemany(
                        "INSERT OR IGNORE INTO _ev_temp VALUES(?,?,?,?,?,?,?,?,?)",
                        [(p,n,e,s,mt,ct,at,d,m) for p,n,e,s,mt,ct,at,d,m in batch]
                    )
                    conn.commit()
                    total_scanned += len(batch)
                    batch.clear()
            if batch:
                cur.executemany(
                    "INSERT OR IGNORE INTO _ev_temp VALUES(?,?,?,?,?,?,?,?,?)",
                    [(p,n,e,s,mt,ct,at,d,m) for p,n,e,s,mt,ct,at,d,m in batch]
                )
                conn.commit()
                total_scanned += len(batch)
            print(f"  {scan_path}: {len(files):,} 文件", flush=True)
    
    t1 = time.time()
    print(f"  扫描总计：{total_scanned:,} 文件 ({t1-t0:.1f}s)", flush=True)

    # ── Phase 2: SQL 集合运算 ──
    print("\n[2] SQL 集合运算...", flush=True)
    now_iso = datetime.now().isoformat()
    conn.execute("PRAGMA synchronous=NORMAL")

    # 2a. INSERT 新增
    t2 = time.time()
    cur.execute("""INSERT INTO files(path,name,ext,size,mtime,ctime,atime,is_dir,depth,scanned_at,is_deleted,file_hash,mime_type,media_meta)
        SELECT path,name,ext,size,mtime,ctime,atime,0,depth,?,0,NULL,mime_type,NULL
        FROM _ev_temp WHERE path NOT IN (SELECT path FROM files)""", (now_iso,))
    inserted = cur.rowcount
    conn.commit()
    t3 = time.time()
    print(f"  新增：{inserted:,} ({t3-t2:.1f}s)", flush=True)

    # 2b. DELETE 多余
    t4 = time.time()
    cur.execute("""CREATE TABLE _to_delete AS
        SELECT f.rowid AS rid FROM files f
        LEFT JOIN _ev_temp e ON f.path = e.path
        WHERE e.path IS NULL""")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM _to_delete")
    to_del = cur.fetchone()[0]
    t5 = time.time()
    print(f"  待删除：{to_del:,} ({t5-t4:.1f}s)", flush=True)

    deleted = 0
    if to_del > 0:
        cur.execute("""DELETE FROM files WHERE rowid IN (SELECT rid FROM _to_delete)""")
        deleted = cur.rowcount
        conn.commit()
        print(f"  已删除：{deleted:,}", flush=True)

    # 2c. UPDATE 变更
    t6 = time.time()
    cur.execute("""UPDATE files SET
        size=ev.size, mtime=ev.mtime, ctime=ev.ctime, atime=ev.atime,
        mime_type=COALESCE(files.mime_type, ev.mime_type)
        FROM _ev_temp ev WHERE files.path = ev.path
        AND (files.size != ev.size OR ABS(files.mtime - ev.mtime) > 2)""")
    updated = cur.rowcount
    conn.commit()
    t7 = time.time()
    print(f"  变更：{updated:,} ({t7-t6:.1f}s)", flush=True)

    # ── Phase 3: 清理 ──
    print("\n[3] 清理临时表...", flush=True)
    conn.execute("PRAGMA synchronous=FULL")
    cur.execute("DROP TABLE IF EXISTS _ev_temp")
    cur.execute("DROP TABLE IF EXISTS _to_delete")
    conn.commit()

    # 最终统计
    cur.execute("SELECT COUNT(*) FROM files")
    total_after = cur.fetchone()[0]
    conn.close()

    db_size = os.path.getsize(DB_PATH) / 1024**2 if os.path.exists(DB_PATH) else 0
    print(f"\n{'='*60}", flush=True)
    print(f"同步完成！总耗时：{time.time()-t0:.1f}s", flush=True)
    print(f"  Before: {total_before:,} → After: {total_after:,}", flush=True)
    print(f"  新增：{inserted:,} | 删除：{deleted:,} | 变更：{updated:,}", flush=True)
    print(f"  数据库：{db_size:.1f} MB", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    sync(dry_run='--dry-run' in sys.argv)