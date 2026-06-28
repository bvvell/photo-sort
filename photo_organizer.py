#!/usr/bin/env python3
"""
photo_organizer.py — Organize and deduplicate a large photo archive safely.

v2 changes over v1:
  - Single batch exiftool call per N files instead of per-file (50-100× faster)
  - Size-grouped hashing: skip files with unique sizes (avoids 80-90% of I/O)
  - Perceptual dedup scoped by year/month (limits O(n²) to same-month cohorts)
  - RAW+JPG pair detection and --keep-raw-jpg-pairs flag
  - --fallback-to-mtime flag: date fallback is now opt-in, not automatic
  - --phash-threshold flag: configurable Hamming distance
  - --resume flag: SQLite-backed state survives interruptions
  - Parallel file hashing via ThreadPoolExecutor (I/O-bound, GIL not a bottleneck)
  - Removed multiprocessing.Pool (batch exiftool makes it unnecessary)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

try:
    from PIL import Image
    import imagehash as _imagehash
    PERCEPTUAL_AVAILABLE = True
except ImportError:
    PERCEPTUAL_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

PHOTO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".heic", ".heif", ".webp", ".avif",
    ".raw", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".ptx", ".pef", ".rw2", ".rwl", ".srw", ".x3f",
    ".raf", ".3fr", ".fff", ".kdc", ".dcr", ".mrw", ".mdc",
}

RAW_EXTENSIONS = {
    ".raw", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".ptx", ".pef", ".rw2", ".rwl", ".srw", ".x3f",
    ".raf", ".3fr", ".fff", ".kdc", ".dcr", ".mrw", ".mdc",
}

HASH_CHUNK = 65536           # 64 KB read window for streaming hash
DEFAULT_PHASH_THRESHOLD = 10 # Hamming distance for pHash match
DEFAULT_BATCH_SIZE = 5_000   # files per exiftool invocation
# Groups larger than this skip perceptual comparison (would be too slow)
PHASH_GROUP_LIMIT = 10_000
STATE_DB_NAME = ".photo_organizer_state.db"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PhotoInfo:
    path: Path
    size: int
    mtime: float
    date_taken: Optional[datetime]
    date_source: str          # "exif" | "mtime" | "unknown"
    exact_hash: Optional[str] = None
    phash: Optional[str] = None
    width: int = 0
    height: int = 0
    is_raw: bool = False

    @property
    def resolution(self) -> int:
        return self.width * self.height


@dataclass
class Stats:
    total: int = 0
    copied: int = 0
    duplicates: int = 0
    raw_jpg_pairs: int = 0
    unknown_date: int = 0
    errors: int = 0
    resumed: int = 0


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("photo_organizer")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "photo_organizer.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class CsvLog:
    FIELDS = ["action", "source", "destination", "reason", "hash", "timestamp"]

    def __init__(self, path: Path, dry_run: bool) -> None:
        self._path = path
        self._dry_run = dry_run
        self._fh = None
        self._writer = None

    def __enter__(self) -> "CsvLog":
        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()
        return self

    def __exit__(self, *_) -> None:
        if self._fh:
            self._fh.close()

    def write(
        self,
        action: str,
        src: Path,
        dst: Optional[Path],
        reason: str = "",
        hash_val: str = "",
    ) -> None:
        if self._dry_run:
            action = f"[DRY-RUN] {action}"
        self._writer.writerow({
            "action": action,
            "source": str(src),
            "destination": str(dst) if dst else "",
            "reason": reason,
            "hash": hash_val,
            "timestamp": datetime.now().isoformat(),
        })
        self._fh.flush()


# ── Resume state ──────────────────────────────────────────────────────────────

class StateDB:
    """Lightweight SQLite store so interrupted runs can be resumed."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS done "
            "(path TEXT PRIMARY KEY, dest TEXT, status TEXT)"
        )
        self._conn.commit()

    def is_done(self, path: Path) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM done WHERE path=?", (str(path),)
            ).fetchone()
            is not None
        )

    def mark_done(self, path: Path, dest: Optional[Path], status: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO done VALUES (?,?,?)",
            (str(path), str(dest) if dest else "", status),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ── File discovery ────────────────────────────────────────────────────────────

def iter_photos(source: Path) -> Iterator[Path]:
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS:
            yield p


# ── Batch EXIF extraction ─────────────────────────────────────────────────────

def exiftool_batch(paths: list[Path]) -> dict[str, dict]:
    """
    Run one exiftool process for an entire batch of files.

    Uses a filelist temp file to avoid ARG_MAX limits on large batches.
    exiftool exits with 1 when individual files produce warnings but still
    emits valid JSON for the files it could read — treat exit code 1 as OK.

    Returns {absolute_path_str: metadata_dict}.
    """
    if not paths:
        return {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(str(p) for p in paths))
        filelist = f.name

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-json",
                "-fast",          # skip MakerNotes (not needed for date/dims)
                "-DateTimeOriginal",
                "-ImageWidth",
                "-ImageHeight",
                "-@", filelist,
            ],
            capture_output=True,
            text=True,
            # No timeout — large batches take time proportional to file count
        )
        if result.returncode not in (0, 1):
            return {}
        data = json.loads(result.stdout)
        return {item["SourceFile"]: item for item in data}
    except (json.JSONDecodeError, KeyError, subprocess.SubprocessError):
        return {}
    finally:
        os.unlink(filelist)


def _make_photo_info(
    path: Path,
    meta: dict,
    fallback_to_mtime: bool,
) -> Optional[PhotoInfo]:
    try:
        stat = path.stat()
    except OSError:
        return None

    raw_date = meta.get("DateTimeOriginal", "")
    if raw_date:
        try:
            date_taken = datetime.strptime(raw_date[:19], "%Y:%m:%d %H:%M:%S")
            date_source = "exif"
        except ValueError:
            # Malformed EXIF date — treat as missing
            date_taken, date_source = None, "unknown"
    elif fallback_to_mtime:
        date_taken = datetime.fromtimestamp(stat.st_mtime)
        date_source = "mtime"
    else:
        date_taken, date_source = None, "unknown"

    return PhotoInfo(
        path=path,
        size=stat.st_size,
        mtime=stat.st_mtime,
        date_taken=date_taken,
        date_source=date_source,
        width=int(meta.get("ImageWidth") or 0),
        height=int(meta.get("ImageHeight") or 0),
        is_raw=path.suffix.lower() in RAW_EXTENSIONS,
    )


def collect_metadata(
    source: Path,
    fallback_to_mtime: bool,
    batch_size: int,
) -> list[PhotoInfo]:
    """
    Discover all photos and extract EXIF in batches via a single exiftool call
    per batch. ~50-100× faster than spawning one exiftool process per file.
    """
    paths = list(iter_photos(source))
    if not paths:
        return []

    results: list[PhotoInfo] = []
    with tqdm(total=len(paths), desc="Scanning metadata", unit="file") as bar:
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]
            meta = exiftool_batch(batch)
            for path in batch:
                info = _make_photo_info(path, meta.get(str(path), {}), fallback_to_mtime)
                if info is not None:
                    results.append(info)
            bar.update(len(batch))

    return results


# ── Hashing ───────────────────────────────────────────────────────────────────

def exact_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> Optional[str]:
    try:
        with Image.open(path) as img:
            return str(_imagehash.phash(img))
    except Exception:
        return None


def compute_exact_hashes_size_grouped(
    photos: list[PhotoInfo], algo: str, workers: int
) -> None:
    """
    Group by file size before hashing.
    Files with a unique size cannot be exact duplicates — skipping them
    typically eliminates 80-90% of hashing work on real photo archives.
    """
    size_groups: dict[int, list[PhotoInfo]] = defaultdict(list)
    for info in photos:
        size_groups[info.size].append(info)

    to_hash = [
        info
        for group in size_groups.values()
        if len(group) > 1
        for info in group
    ]
    skipped = len(photos) - len(to_hash)
    if skipped:
        tqdm.write(f"  Size filter: skipping {skipped} unique-size files")
    if not to_hash:
        return

    def _hash_one(info: PhotoInfo) -> None:
        try:
            info.exact_hash = exact_hash(info.path, algo)
        except OSError:
            info.exact_hash = None

    if workers > 1:
        # Hashing is I/O-bound; threads work fine here (GIL released on read)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_hash_one, info): info for info in to_hash}
            with tqdm(total=len(to_hash), desc="Hashing files", unit="file") as bar:
                for fut in as_completed(futures):
                    fut.result()
                    bar.update(1)
    else:
        for info in tqdm(to_hash, desc="Hashing files", unit="file"):
            _hash_one(info)


def compute_perceptual_hashes(photos: list[PhotoInfo]) -> None:
    for info in tqdm(photos, desc="Perceptual hashing", unit="file"):
        info.phash = perceptual_hash(info.path)


# ── Duplicate detection ───────────────────────────────────────────────────────

def _score(info: PhotoInfo) -> tuple:
    """Higher = better candidate to keep (RAW > resolution > size)."""
    return (int(info.is_raw), info.resolution, info.size)


def detect_exact_duplicates(photos: list[PhotoInfo]) -> dict[str, list[PhotoInfo]]:
    groups: dict[str, list[PhotoInfo]] = defaultdict(list)
    for p in photos:
        if p.exact_hash:
            groups[p.exact_hash].append(p)
    return {k: v for k, v in groups.items() if len(v) > 1}


def detect_perceptual_duplicates(
    photos: list[PhotoInfo],
    threshold: int,
) -> list[list[PhotoInfo]]:
    """
    Group by year/month before O(n²) pHash comparison.

    Genuine perceptual duplicates (burst shots, re-exports) almost always
    occur within the same month. Scoping comparison to monthly cohorts keeps
    worst-case work proportional to the busiest single month, not the whole
    archive. Groups larger than PHASH_GROUP_LIMIT are skipped with a warning.
    """
    buckets: dict[tuple, list[PhotoInfo]] = defaultdict(list)
    for p in photos:
        key = (p.date_taken.year, p.date_taken.month) if p.date_taken else (0, 0)
        buckets[key].append(p)

    all_groups: list[list[PhotoInfo]] = []
    for (year, month), bucket in buckets.items():
        if len(bucket) > PHASH_GROUP_LIMIT:
            label = f"{year}/{month:02d}" if year else "unknown-date"
            tqdm.write(
                f"  WARNING: {label} has {len(bucket)} files — "
                f"skipping perceptual dedup for this month (>{PHASH_GROUP_LIMIT} limit)"
            )
            continue
        all_groups.extend(_phash_within_group(bucket, threshold))
    return all_groups


def _phash_within_group(
    photos: list[PhotoInfo], threshold: int
) -> list[list[PhotoInfo]]:
    candidates = [p for p in photos if p.phash]
    groups: list[list[PhotoInfo]] = []
    used: set[int] = set()
    for i, a in enumerate(candidates):
        if i in used:
            continue
        ha = _imagehash.hex_to_hash(a.phash)
        group = [a]
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            if ha - _imagehash.hex_to_hash(candidates[j].phash) <= threshold:
                group.append(candidates[j])
                used.add(j)
        if len(group) > 1:
            used.add(i)
            groups.append(group)
    return groups


# ── RAW + JPG pair detection ──────────────────────────────────────────────────

def find_raw_jpg_pairs(photos: list[PhotoInfo]) -> dict[str, list[PhotoInfo]]:
    """
    Detect files sharing the same stem but with mixed RAW/non-RAW extensions.
    Example: IMG_1234.CR2 + IMG_1234.JPG → one group keyed on "img_1234".

    Only returns stems that have at least one RAW and at least one non-RAW file,
    so pure duplicates (two JPGs) are handled by the normal hash path instead.
    """
    by_stem: dict[str, list[PhotoInfo]] = defaultdict(list)
    for p in photos:
        by_stem[p.path.stem.lower()].append(p)

    return {
        stem: files
        for stem, files in by_stem.items()
        if len(files) > 1
        and any(f.is_raw for f in files)
        and any(not f.is_raw for f in files)
    }


# ── Dedup map builder ─────────────────────────────────────────────────────────

def build_dup_map(
    photos: list[PhotoInfo],
    deduplicate: bool,
    perceptual: bool,
    keep_raw_jpg_pairs: bool,
    phash_threshold: int,
) -> tuple[dict[Path, str], dict[Path, str]]:
    """
    Decide the fate of every file.

    Returns:
      dup_flags   — {path: "dup"} for files going to _duplicates_review
      dup_reasons — {path: reason_str} for the CSV log

    Evaluation order (first match wins):
      1. RAW+JPG pairs  — explicit basename grouping
      2. Exact hashes   — byte-for-byte identical files
      3. Perceptual     — visually similar images
    """
    dup_flags: dict[Path, str] = {}
    dup_reasons: dict[Path, str] = {}

    def mark_dup(path: Path, reason: str) -> None:
        dup_flags[path] = "dup"
        dup_reasons[path] = reason

    # 1. RAW+JPG pairs
    if not keep_raw_jpg_pairs:
        for _stem, group in find_raw_jpg_pairs(photos).items():
            raw_files = [f for f in group if f.is_raw]
            non_raw   = [f for f in group if not f.is_raw]
            # Keep the best RAW; send extra RAWs (unusual) and all non-RAW to review
            keeper = max(raw_files, key=_score)
            for f in raw_files:
                if f is not keeper:
                    mark_dup(f.path, "raw_duplicate")
            for f in non_raw:
                mark_dup(f.path, "raw_jpg_pair")

    # 2. Exact duplicates
    if deduplicate:
        for group in detect_exact_duplicates(photos).values():
            keeper = max(group, key=_score)
            for p in group:
                if p.path not in dup_flags and p is not keeper:
                    mark_dup(p.path, "exact_hash")

    # 3. Perceptual duplicates
    if perceptual and PERCEPTUAL_AVAILABLE:
        for group in detect_perceptual_duplicates(photos, phash_threshold):
            keeper = max(group, key=_score)
            for p in group:
                if p.path not in dup_flags and p is not keeper:
                    mark_dup(p.path, "perceptual_hash")

    return dup_flags, dup_reasons


# ── Destination helpers ───────────────────────────────────────────────────────

def target_path(info: PhotoInfo, output: Path) -> Path:
    if info.date_taken is None:
        folder = output / "unknown"
    else:
        dt = info.date_taken
        folder = output / f"{dt.year:04d}" / f"{dt.month:02d}"
    return folder / info.path.name


def safe_dest(path: Path) -> Path:
    """Append _1, _2, … until a non-existing name is found."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


# ── File operations ───────────────────────────────────────────────────────────

def do_copy(src: Path, dst: Path, dry_run: bool) -> Path:
    actual = safe_dest(dst)
    if not dry_run:
        actual.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, actual)
    return actual


def do_copy_to_dups(src: Path, dup_dir: Path, dry_run: bool) -> Path:
    dest = safe_dest(dup_dir / src.name)
    if not dry_run:
        dup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process(
    photos: list[PhotoInfo],
    output: Path,
    deduplicate: bool,
    perceptual: bool,
    keep_raw_jpg_pairs: bool,
    phash_threshold: int,
    dry_run: bool,
    csv_log: CsvLog,
    logger: logging.Logger,
    stats: Stats,
    state_db: Optional[StateDB],
) -> None:
    dup_dir = output / "_duplicates_review"
    dup_flags, dup_reasons = build_dup_map(
        photos, deduplicate, perceptual, keep_raw_jpg_pairs, phash_threshold
    )
    for info in tqdm(photos, desc="Copying files", unit="file"):
        try:
            _process_one(
                info, output, dup_dir, dup_flags, dup_reasons,
                dry_run, csv_log, logger, stats, state_db,
            )
        except Exception as exc:
            stats.errors += 1
            logger.error("Error processing %s: %s", info.path, exc)
            csv_log.write("ERROR", info.path, None, reason=str(exc))


def _process_one(
    info: PhotoInfo,
    output: Path,
    dup_dir: Path,
    dup_flags: dict[Path, str],
    dup_reasons: dict[Path, str],
    dry_run: bool,
    csv_log: CsvLog,
    logger: logging.Logger,
    stats: Stats,
    state_db: Optional[StateDB],
) -> None:
    if state_db and state_db.is_done(info.path):
        stats.resumed += 1
        return

    is_dup = info.path in dup_flags
    reason = dup_reasons.get(info.path, "")

    if is_dup:
        dest = do_copy_to_dups(info.path, dup_dir, dry_run)
        stats.duplicates += 1
        if reason == "raw_jpg_pair":
            stats.raw_jpg_pairs += 1
        logger.debug("DUP  %s → %s (%s)", info.path.name, dest, reason)
        csv_log.write("DUPLICATE_REVIEW", info.path, dest, reason=reason, hash_val=info.exact_hash or "")
    else:
        base = target_path(info, output)
        dest = do_copy(info.path, base, dry_run)
        stats.copied += 1
        if info.date_taken is None:
            stats.unknown_date += 1
        logger.debug("COPY %s → %s", info.path.name, dest)
        csv_log.write("COPY", info.path, dest, reason=info.date_source, hash_val=info.exact_hash or "")

    if state_db and not dry_run:
        state_db.mark_done(info.path, dest, "done")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="photo-organizer",
        description="Organize and deduplicate a photo archive safely.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  photo-organizer --source /input --output /output
  photo-organizer --source /input --output /output --deduplicate --dry-run
  photo-organizer --source /input --output /output --deduplicate --perceptual --workers 8
  photo-organizer --source /input --output /output --fallback-to-mtime --resume
  photo-organizer --source /input --output /output --keep-raw-jpg-pairs
""",
    )
    p.add_argument("--source",             required=True, type=Path, metavar="DIR")
    p.add_argument("--output",             required=True, type=Path, metavar="DIR")
    p.add_argument("--deduplicate",        action="store_true",
                   help="Detect and segregate exact duplicates")
    p.add_argument("--perceptual",         action="store_true",
                   help="Also detect visually similar images via pHash")
    p.add_argument("--dry-run",            action="store_true",
                   help="Preview actions without writing any files")
    p.add_argument("--fallback-to-mtime",  action="store_true",
                   help="Use file mtime when EXIF date is absent "
                        "(default: send to /unknown/ instead)")
    p.add_argument("--keep-raw-jpg-pairs", action="store_true",
                   help="Copy both RAW and JPG when they share a basename "
                        "(default: keep RAW, send JPG to _duplicates_review)")
    p.add_argument("--hash-algorithm",     default="sha256", choices=["md5", "sha256"])
    p.add_argument("--phash-threshold",    type=int, default=DEFAULT_PHASH_THRESHOLD,
                   metavar="N",
                   help=f"Max Hamming distance for perceptual match "
                        f"(default: {DEFAULT_PHASH_THRESHOLD})")
    p.add_argument("--batch-size",         type=int, default=DEFAULT_BATCH_SIZE,
                   metavar="N",
                   help=f"Files per exiftool batch call "
                        f"(default: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--workers",            type=int, default=4, metavar="N",
                   help="Threads for parallel file hashing (default: 4)")
    p.add_argument("--resume",             action="store_true",
                   help="Skip files already processed in a previous run")
    p.add_argument("--verbose",            action="store_true")
    return p


def check_deps(perceptual: bool) -> None:
    try:
        subprocess.run(["exiftool", "-ver"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit(
            "ERROR: exiftool not found.\n"
            "  macOS:  brew install exiftool\n"
            "  Debian: apt install libimage-exiftool-perl"
        )
    if perceptual and not PERCEPTUAL_AVAILABLE:
        sys.exit(
            "ERROR: --perceptual requires Pillow and imagehash.\n"
            "  pip install Pillow imagehash"
        )


def main() -> None:
    args = build_parser().parse_args()
    check_deps(args.perceptual)

    if not args.source.is_dir():
        sys.exit(f"ERROR: source not found: {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(args.output, args.verbose)
    stats = Stats()

    if args.dry_run:
        print("[DRY-RUN] No files will be written.")

    state_db: Optional[StateDB] = None
    if args.resume:
        db_path = args.output / STATE_DB_NAME
        state_db = StateDB(db_path)
        print(f"Resume mode active — state: {db_path}")

    csv_path = args.output / "log.csv"

    try:
        with CsvLog(csv_path, dry_run=args.dry_run) as csv_log:

            # Phase 1 — batch EXIF scan
            print(f"Scanning {args.source} …")
            photos = collect_metadata(
                args.source,
                fallback_to_mtime=args.fallback_to_mtime,
                batch_size=args.batch_size,
            )
            stats.total = len(photos)
            print(f"Found {stats.total} photos.")
            if not photos:
                return

            # Phase 2 — size-grouped exact hashing
            if args.deduplicate:
                compute_exact_hashes_size_grouped(
                    photos, args.hash_algorithm, args.workers
                )

            # Phase 3 — perceptual hashing
            if args.perceptual:
                compute_perceptual_hashes(photos)

            # Phase 4 — copy / deduplicate
            process(
                photos=photos,
                output=args.output,
                deduplicate=args.deduplicate,
                perceptual=args.perceptual,
                keep_raw_jpg_pairs=args.keep_raw_jpg_pairs,
                phash_threshold=args.phash_threshold,
                dry_run=args.dry_run,
                csv_log=csv_log,
                logger=logger,
                stats=stats,
                state_db=state_db,
            )
    finally:
        if state_db:
            state_db.close()

    _print_summary(stats, csv_path, args.dry_run)


def _print_summary(stats: Stats, csv_path: Path, dry_run: bool) -> None:
    print("\n── Summary " + "─" * 38)
    print(f"  Total scanned   : {stats.total}")
    print(f"  Copied          : {stats.copied}")
    print(f"  Duplicates sent : {stats.duplicates}")
    if stats.raw_jpg_pairs:
        print(f"    RAW+JPG pairs : {stats.raw_jpg_pairs}")
    print(f"  Unknown date    : {stats.unknown_date}")
    if stats.resumed:
        print(f"  Resumed (skip)  : {stats.resumed}")
    print(f"  Errors          : {stats.errors}")
    print(f"  Log             : {csv_path}")
    if dry_run:
        print("  [DRY-RUN — no files were modified]")


if __name__ == "__main__":
    main()
