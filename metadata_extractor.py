#!/usr/bin/env python3
"""
元数据提取器 — 跨平台版本
- 图片：宽高/格式/EXIF (PIL)
- 视频/音频：时长/码率/分辨率/编解码器 (pymediainfo+TinyTag)
- PDF：文字层检测/页数 (PyMuPDF)
- 支持多盘并行处理
"""

import sqlite3
import os
import threading
import time
import queue
import json
import signal
import sys
import traceback
import platform
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

# 配置加载
try:
    from config_loader import get_config, get_env
    config = get_config()
    metadata_config = config.get('metadata', {})
    DB_PATH = config.get('database', {}).get('path', './file_index.db')
    DISK_WORKERS = metadata_config.get('workers_per_disk', 4)
    FILE_TIMEOUT = metadata_config.get('file_timeout', 20)
    LOG_FILE = metadata_config.get('log_file', './logs/meta_extract.log')
    MAX_PENDING = metadata_config.get('max_pending', 200)
except ImportError:
    DB_PATH = './file_index.db'
    DISK_WORKERS = 4
    FILE_TIMEOUT = 20
    LOG_FILE = './logs/meta_extract.log'
    MAX_PENDING = 200

# 常量
GET_TIMEOUT = 25
BATCH_WRITE = 5000
WRITE_QUEUE_MAX = 200000
COMMIT_INTERVAL = 3

# 文件类型扩展名
SKIP_EXTS = {
    '.svg', '.xpm', '.djvu', '.psd', '.ai', '.eps', '.xcf', '.bmp',
    '.tga', '.rgb', '.pgm', '.ppm', '.pbm', '.pnm', '.dds',
    '.m3u', '.m3u8', '.pls', '.cue', '.txt', '.ini', '.log',
}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif', '.ico', '.heic', '.heif', '.avif'}
MEDIA_EXTS = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma',
    '.opus', '.wv', '.ape', '.alac', '.amr', '.mid', '.midi',
    '.3gp', '.asf', '.bik', '.mpeg', '.mpg', '.m4v', '.rm', '.rmvb',
    '.ts', '.mts', '.m2ts', '.wem',
}
AUDIO_ONLY_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.opus', '.wv', '.ape', '.alac', '.amr', '.mid', '.midi'}
PDF_EXT = '.pdf'

# 全局变量
stats = {'done': 0, 'errors': 0, 'total': 0, 'skipped': 0,
         'image': 0, 'media': 0, 'pdf_text': 0, 'pdf_image': 0, 'timeout': 0}
stats_lock = threading.Lock()
start_time = datetime.now()
write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAX)
stop_signal = threading.Event()
shutdown_requested = False
MAIN_PID = os.getpid()

# 日志
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    stop_signal.set()
    if os.getpid() == MAIN_PID:
        log(f"收到信号 {signum}，优雅退出...")

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# 图片处理
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1_000_000_000
except ImportError:
    Image = None
    log("警告: PIL 未安装，图片元数据提取将跳过")

# 音频/视频处理
try:
    from tinytag import TinyTag
except ImportError:
    TinyTag = None
    log("警告: TinyTag 未安装，音频元数据提取将跳过")

try:
    import pymediainfo
except ImportError:
    pymediainfo = None
    log("警告: pymediainfo 未安装，视频元数据提取将跳过")

# PDF处理
try:
    import fitz
except ImportError:
    fitz = None
    log("警告: PyMuPDF 未安装，PDF元数据提取将跳过")


def get_ext(path):
    dot = path.rfind('.')
    return path[dot:].lower() if dot >= 0 else ''


def extract_image_meta(path):
    """提取图片元数据"""
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            meta = {'type': 'image', 'width': img.width, 'height': img.height,
                    'format': img.format or '', 'mode': img.mode or ''}
            try:
                exif = img.getexif()
                if exif:
                    for tid, key in [(0x0112, 'orientation'), (0x010F, 'make'),
                                     (0x0110, 'model'), (0x9003, 'datetime_original'),
                                     (0x829A, 'exposure_time'), (0x829D, 'f_number'),
                                     (0x8827, 'iso'), (0x920A, 'focal_length')]:
                        v = exif.get(tid)
                        if v is not None:
                            meta[key] = str(v)
            except Exception:
                pass
            return meta
    except Exception:
        return None


def extract_audio_meta(path):
    """提取音频元数据"""
    if TinyTag is None:
        return None
    try:
        tag = TinyTag.get(path)
        meta = {'type': 'audio'}
        if tag.duration:
            meta['duration'] = round(tag.duration, 3)
        if tag.bitrate:
            meta['bitrate'] = int(tag.bitrate)
        if tag.samplerate:
            meta['samplerate'] = int(tag.samplerate)
        if tag.channels:
            meta['channels'] = int(tag.channels)
        if tag.artist:
            meta['artist'] = tag.artist
        if tag.title:
            meta['title'] = tag.title
        if tag.album:
            meta['album'] = tag.album
        if tag.year:
            meta['year'] = str(tag.year)
        if tag.genre:
            meta['genre'] = tag.genre
        return meta if len(meta) > 1 else None
    except Exception:
        return None


def extract_video_meta(path):
    """提取视频元数据"""
    if pymediainfo is None:
        return None
    try:
        mi = pymediainfo.MediaInfo.parse(path)
        meta = {'type': 'video'}
        if mi.general_tracks:
            g = mi.general_tracks[0]
            if g.duration:
                meta['duration'] = round(float(g.duration), 3)
            if g.format:
                meta['format'] = g.format
            if g.bit_rate:
                meta['bit_rate'] = int(g.bit_rate)
        for t in mi.video_tracks:
            v = {}
            if t.width:
                v['width'] = t.width
            if t.height:
                v['height'] = t.height
            if t.frame_rate:
                v['frame_rate'] = float(t.frame_rate)
            if t.codec_id:
                v['codec'] = t.codec_id
            if v:
                meta['video'] = v
            break
        for t in mi.audio_tracks:
            a = {}
            if t.sample_rate:
                a['sample_rate'] = int(t.sample_rate)
            if t.channels:
                a['channels'] = t.channels
            if t.codec_id:
                a['codec'] = t.codec_id
            if a:
                meta['audio'] = a
            break
        return meta if len(meta) > 1 else None
    except Exception:
        return None


def extract_pdf_meta(path):
    """提取PDF元数据"""
    if fitz is None:
        return None
    try:
        if not os.path.exists(path):
            return None
        doc = fitz.open(path)
        page_count = len(doc)
        has_text = False
        check_pages = min(page_count, 3)
        text_pages = 0
        for i in range(check_pages):
            page = doc[i]
            text = page.get_text("text", sort=True).strip()
            if len(text) > 20:
                text_pages += 1
                has_text = True
        doc.close()
        text_ratio = text_pages / check_pages if check_pages > 0 else 0
        return {
            'type': 'text_pdf' if has_text else 'image_pdf',
            'has_text': has_text,
            'page_count': page_count,
            'text_ratio': round(text_ratio, 2),
            'format': 'PDF',
        }
    except fitz.FileDataError:
        return None
    except Exception:
        return None


# 跨平台超时处理
if platform.system() == 'Windows':
    # Windows: 使用 threading 超时，不依赖 SIGALRM
    def timeout_handler(signum, frame):
        raise TimeoutError('file processing timeout')
    
    def process_file_with_timeout(args):
        """Windows 版：使用 threading 超时"""
        fid, path, mime_type = args
        ext = get_ext(path)
        if ext in SKIP_EXTS:
            return (fid, None, 'skip')
        
        result_holder = [None, None, None]  # [meta, category, error]
        
        def _process():
            try:
                if ext == PDF_EXT:
                    meta = extract_pdf_meta(path)
                    result_holder[0] = meta
                    result_holder[1] = 'pdf_text' if (meta and meta.get('has_text')) else 'pdf_image'
                elif ext in IMAGE_EXTS or (mime_type and mime_type.startswith('image/')):
                    meta = extract_image_meta(path)
                    result_holder[0] = meta
                    result_holder[1] = 'image'
                elif ext in AUDIO_ONLY_EXTS or (mime_type and mime_type == 'audio/mpeg'):
                    meta = extract_audio_meta(path)
                    result_holder[0] = meta
                    result_holder[1] = 'media'
                elif ext in MEDIA_EXTS or (mime_type and (mime_type.startswith('video/') or mime_type.startswith('audio/'))):
                    if ext not in AUDIO_ONLY_EXTS:
                        meta = extract_video_meta(path)
                    else:
                        meta = extract_audio_meta(path)
                    result_holder[0] = meta
                    result_holder[1] = 'media'
                else:
                    result_holder[1] = 'skip'
            except Exception as e:
                result_holder[2] = e
        
        t = threading.Thread(target=_process, daemon=True)
        t.start()
        t.join(timeout=FILE_TIMEOUT)
        
        if t.is_alive():
            return (fid, 'TIMEOUT', 'timeout')
        if result_holder[2]:
            return (fid, 'FAIL', 'error')
        if result_holder[0]:
            return (fid, result_holder[0], result_holder[1])
        return (fid, None, result_holder[1] or 'skip')

else:
    # Linux/Mac: 使用 SIGALRM
    def timeout_handler(signum, frame):
        raise TimeoutError('file processing timeout')
    
    def process_file(args):
        """Linux/Mac 版：使用 SIGALRM 超时"""
        fid, path, mime_type = args
        ext = get_ext(path)
        if ext in SKIP_EXTS:
            return (fid, None, 'skip')
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(FILE_TIMEOUT)
        
        try:
            if ext == PDF_EXT:
                meta = extract_pdf_meta(path)
                signal.alarm(0)
                return (fid, meta, 'pdf_text' if (meta and meta.get('has_text')) else 'pdf_image') if meta else (fid, 'FAIL', 'error')
            if ext in IMAGE_EXTS or (mime_type and mime_type.startswith('image/')):
                meta = extract_image_meta(path)
                signal.alarm(0)
                return (fid, meta, 'image') if meta else (fid, 'FAIL', 'error')
            if ext in AUDIO_ONLY_EXTS or (mime_type and mime_type == 'audio/mpeg'):
                meta = extract_audio_meta(path)
                signal.alarm(0)
                return (fid, meta, 'media') if meta else (fid, 'FAIL', 'error')
            if ext in MEDIA_EXTS or (mime_type and (mime_type.startswith('video/') or mime_type.startswith('audio/'))):
                if ext not in AUDIO_ONLY_EXTS:
                    meta = extract_video_meta(path)
                else:
                    meta = extract_audio_meta(path)
                signal.alarm(0)
                return (fid, meta, 'media') if meta else (fid, 'FAIL', 'error')
            signal.alarm(0)
            return (fid, None, 'skip')
        except TimeoutError:
            signal.alarm(0)
            return (fid, 'TIMEOUT', 'timeout')
        except Exception:
            signal.alarm(0)
            return (fid, 'FAIL', 'error')
    
    process_file_with_timeout = process_file


def _init_worker():
    """子进程初始化"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, timeout_handler)


def writer_thread():
    """数据库写入线程"""
    conn = sqlite3.connect(DB_PATH, timeout=300, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1024000")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    cur = conn.cursor()
    buffer = []
    last_commit = time.time()

    while not stop_signal.is_set() or not write_queue.empty():
        try:
            item = write_queue.get(timeout=0.3)
            fid, meta, category = item
            if meta:
                buffer.append((json.dumps(meta, ensure_ascii=False) if meta != 'FAIL' else 'FAIL', fid))
            now = time.time()
            if len(buffer) >= BATCH_WRITE or (buffer and now - last_commit >= COMMIT_INTERVAL):
                cur.execute("BEGIN")
                cur.executemany("UPDATE files SET media_meta=? WHERE id=?", buffer)
                cur.execute("COMMIT")
                with stats_lock:
                    stats['done'] += len(buffer)
                buffer.clear()
                last_commit = now
        except queue.Empty:
            now = time.time()
            if buffer and now - last_commit >= COMMIT_INTERVAL:
                cur.execute("BEGIN")
                cur.executemany("UPDATE files SET media_meta=? WHERE id=?", buffer)
                cur.execute("COMMIT")
                with stats_lock:
                    stats['done'] += len(buffer)
                buffer.clear()
                last_commit = now
            if stop_signal.is_set():
                break

    if buffer:
        try:
            cur.execute("BEGIN")
            cur.executemany("UPDATE files SET media_meta=? WHERE id=?", buffer)
            cur.execute("COMMIT")
            with stats_lock:
                stats['done'] += len(buffer)
        except Exception:
            pass
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        pass
    conn.close()


def process_disk(disk, files, workers):
    """处理单个磁盘的文件"""
    total = len(files)
    log(f"[{disk.upper()}] 启动: {total:,} 文件, {workers} workers")

    pool = mp.Pool(workers, initializer=_init_worker)
    submitted = 0
    collected = 0
    pending = []

    try:
        while (submitted < total or pending) and not shutdown_requested:
            # 提交新任务
            while submitted < total and len(pending) < MAX_PENDING:
                ar = pool.apply_async(process_file_with_timeout, (files[submitted],))
                pending.append((submitted, ar))
                submitted += 1

            # 收集结果
            still_pending = []
            for idx, ar in pending:
                if ar.ready():
                    try:
                        result = ar.get(timeout=1)
                        fid, meta, category = result
                        if meta and meta not in ('FAIL', 'TIMEOUT'):
                            write_queue.put((fid, meta, category), timeout=5)
                            with stats_lock:
                                stats[category] = stats.get(category, 0) + 1
                        elif meta == 'TIMEOUT':
                            write_queue.put((fid, 'TIMEOUT', 'timeout'), timeout=5)
                            with stats_lock:
                                stats['timeout'] += 1
                        elif meta == 'FAIL':
                            write_queue.put((fid, 'FAIL', 'error'), timeout=5)
                            with stats_lock:
                                stats['errors'] += 1
                        else:
                            with stats_lock:
                                if category == 'skip':
                                    stats['skipped'] += 1
                                else:
                                    stats['errors'] += 1
                        collected += 1
                    except Exception:
                        with stats_lock:
                            stats['errors'] += 1
                        collected += 1
                else:
                    still_pending.append((idx, ar))
            pending = still_pending

            # 进度日志
            if collected % 5000 < workers or (not pending and submitted >= total):
                with stats_lock:
                    d = stats['done']
                    e = stats['errors']
                    s = stats['skipped']
                log(f"[{disk}] {collected:,}/{total:,} done={d:,} err={e:,} skip={s:,} pending={len(pending)}")

            if not pending and submitted >= total:
                break
            time.sleep(0.05)

    except Exception as e:
        log(f"[{disk}] 异常: {e}")
    finally:
        pool.terminate()
        pool.join()

    log(f"[{disk.upper()}] ✅ 完成: {collected:,}/{total:,}")


def reporter():
    """定期进度报告"""
    while not stop_signal.is_set():
        time.sleep(15)
        with stats_lock:
            d = stats['done']
            e = stats['errors']
            s = stats['skipped']
            t = stats['total']
            to = stats['timeout']
        elapsed = (datetime.now() - start_time).total_seconds()
        pct = d * 100 / t if t > 0 else 0
        speed = d / elapsed if elapsed > 0 else 0
        eta = (t - d) / speed / 60 if speed > 0 else 0
        log(f"{d:,}/{t:,} ({pct:.1f}%) | {speed:.0f}/s | ETA:{eta:.0f}m | 错误:{e:,} 跳过:{s:,}")


def extract_metadata():
    """主入口函数"""
    log("=" * 60)
    log("🚀 元数据提取器 (跨平台版)")
    log("=" * 60)

    try:
        conn = sqlite3.connect(DB_PATH, timeout=300)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, path, mime_type FROM files
            WHERE is_deleted=0
            AND (mime_type LIKE 'image/%' OR mime_type LIKE 'video/%'
                 OR mime_type LIKE 'audio/%' OR ext = '.pdf')
            AND (media_meta IS NULL OR media_meta = '')
        """)
        all_files = cur.fetchall()
        conn.close()
    except Exception as e:
        log(f"DB 错误: {e}")
        sys.exit(1)

    total = len(all_files)
    stats['total'] = total
    log(f"待处理: {total:,}")

    if total == 0:
        log("全部完成!")
        return

    # 按磁盘分组
    disk_files = {}
    for fid, path, mime in all_files:
        if path.startswith('/mnt/'):
            disk = path[5].lower()
        elif path.startswith('/home/') or path.startswith('/root/') or path.startswith('C:\\'):
            disk = '_home'
        else:
            # 使用路径前缀作为 disk
            parts = path.split(os.sep)
            if len(parts) > 1:
                disk = parts[1] if parts[0] == '' else parts[0]
            else:
                disk = '_default'
        disk_files.setdefault(disk, []).append((fid, path, mime))

    for disk in sorted(disk_files):
        cnt = len(disk_files[disk])
        log(f"  {disk.upper()}: {cnt:,} 文件")

    # 启动写入线程 + 报告线程
    writer = threading.Thread(target=writer_thread, daemon=True)
    writer.start()
    threading.Thread(target=reporter, daemon=True).start()

    # 各盘并行处理
    threads = []
    for disk, files in disk_files.items():
        workers = min(DISK_WORKERS, len(files))
        t = threading.Thread(target=process_disk, args=(disk, files, workers))
        t.start()
        threads.append(t)
        time.sleep(0.5)

    for t in threads:
        t.join()

    stop_signal.set()
    writer.join(timeout=60)

    elapsed = (datetime.now() - start_time).total_seconds()
    with stats_lock:
        log(f"✅ 全部完成! 成功:{stats['done']:,} 图片:{stats['image']:,} "
            f"媒体:{stats['media']:,} PDF文字:{stats['pdf_text']:,} "
            f"PDF图片:{stats['pdf_image']:,} 错误:{stats['errors']:,} "
            f"跳过:{stats['skipped']:,} | {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == '__main__':
    try:
        extract_metadata()
    except Exception as e:
        log(f"异常: {e}\n{traceback.format_exc()}")
        sys.exit(1)