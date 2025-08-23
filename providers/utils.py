from __future__ import annotations

import hashlib
import threading
from queue import Queue
from typing import Any, Callable, Optional, Tuple

from PIL import Image

import os
import pickle
import time

CACHE_DIR = os.path.join(os.getcwd(), ".cache", "image_understanding")
CACHE_CAP_BYTES = int(os.environ.get("IMAGE_CACHE_CAP_BYTES", str(2 * 1024 * 1024 * 1024)))  # default 2GB
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_image_hash(image: Image.Image) -> str:
    img_bytes = image.tobytes()
    h = hashlib.sha1()
    h.update(img_bytes)
    h.update(str(image.size).encode("utf-8"))
    return h.hexdigest()


def resize_for_processing(image: Image.Image, max_long_side: int = 1600) -> Tuple[Image.Image, float, float]:
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return image.copy(), 1.0, 1.0
    scale = max_long_side / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    sx = w / float(new_w)
    sy = h / float(new_h)
    return resized, sx, sy


def run_with_timeout(func: Callable[[], Any], timeout_sec: float, default: Any = None) -> Any:
    q: Queue = Queue(maxsize=1)

    def _runner():
        try:
            q.put(func())
        except Exception as e:
            q.put(e)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return default
    result = q.get()
    if isinstance(result, Exception):
        return default
    return result


def _cache_path(key: str) -> str:
    safe = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def disk_cache_read(key: str) -> Optional[Any]:
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # touch for LRU
        os.utime(path, None)
        return data
    except Exception:
        return None


def _enforce_lru_cap():
    try:
        entries = []
        total = 0
        for fname in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, fname)
            try:
                st = os.stat(p)
                total += st.st_size
                entries.append((st.st_mtime, st.st_size, p))
            except Exception:
                continue
        if total <= CACHE_CAP_BYTES:
            return
        entries.sort()  # oldest first
        for _, size, p in entries:
            try:
                os.remove(p)
                total -= size
                if total <= CACHE_CAP_BYTES:
                    break
            except Exception:
                continue
    except Exception:
        pass


def disk_cache_write(key: str, value: Any) -> None:
    path = _cache_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        _enforce_lru_cap()
    except Exception:
        pass


def disk_cache_clear() -> None:
    try:
        for fname in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, fname)
            try:
                os.remove(p)
            except Exception:
                continue
    except Exception:
        pass


def disk_cache_size_bytes() -> int:
    total = 0
    try:
        for fname in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, fname)
            try:
                total += os.stat(p).st_size
            except Exception:
                continue
    except Exception:
        return 0
    return total 