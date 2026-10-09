"""Fetch only the tabular files of the CGMacros dataset (PhysioNet, open access, CC BY-NC-SA 4.0).

The published zip is ~650 MB because of meal photographs. This script reads the zip's central
directory over HTTP range requests and extracts just the CSVs (~a few % of the archive).

    python scripts/fetch_cgmacros.py

Citation: Gutierrez-Osuna R., Kerr D., Mortazavi B., Das A. CGMacros: a scientific dataset for
personalized nutrition and diet monitoring (version 1.0.0). PhysioNet (2025).
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "external" / "cgmacros"


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file backed by HTTP range requests, with a block cache."""

    BLOCK = 1 << 16   # small blocks: the server throttles each connection, so parallelism wins

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()

    def prefetch(self, start: int, end: int, workers: int = 32) -> None:
        """Fetch a byte range in parallel blocks (the server is slow per connection)."""
        blocks = [b for b in range(start // self.BLOCK, (end - 1) // self.BLOCK + 1) if b not in self.cache]
        def get(b):
            s = b * self.BLOCK
            return b, self._fetch(s, min(s + self.BLOCK, self.size))
        with ThreadPoolExecutor(workers) as ex:
            for b, data in ex.map(get, blocks):
                self.cache[b] = data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        self.prefetch(self.pos, self.pos + n)
        out = bytearray()
        p = self.pos
        while len(out) < n:
            b, off = divmod(p, self.BLOCK)
            chunk = self.cache[b][off: off + n - len(out)]
            out += chunk
            p += len(chunk)
        self.pos = p
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = HttpRangeFile(URL)
    print(f"archive size {f.size / 1e6:.0f} MB; reading central directory ...")
    f.prefetch(f.size - 520_000, f.size)  # central directory is ~480 KB
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    csvs = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
    total = sum(i.compress_size for i in csvs)
    print(f"{len(csvs)} CSV files, {total / 1e6:.1f} MB compressed")
    for k, info in enumerate(csvs, 1):
        dest = OUT / Path(info.filename).name
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue
        # header (30 bytes + name + extra) precedes the data; prefetch generously in parallel
        f.prefetch(info.header_offset, min(info.header_offset + info.compress_size + 1024, f.size))
        dest.write_bytes(zf.read(info))
        print(f"  [{k}/{len(csvs)}] {info.filename} ({info.file_size / 1e6:.1f} MB)")
        sys.stdout.flush()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
