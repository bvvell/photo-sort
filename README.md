# photo-sort

> Built with [Claude Code](https://claude.ai/code) (Anthropic)

Two CLI tools for working with large photo archives:

| Tool | What it does |
|---|---|
| `photo_organizer.py` | Organizes and deduplicates an archive into a date tree |
| `select_best_shots.py` | Scores images by quality and copies the top picks |

**Safety guarantee:** originals are never deleted or overwritten. Everything is copy-first.

---

## Install

```bash
# System dependency (photo_organizer.py only)
brew install exiftool          # macOS
# apt install libimage-exiftool-perl   # Debian/Ubuntu

# Core Python deps
pip install -r requirements.txt

# Full deps — adds RAW support + better metrics for select_best_shots.py
pip install -r requirements-full.txt
```

**Optional extras (included in requirements-full.txt):**

| Package | Adds |
|---|---|
| `rawpy` | RAW file scoring (CR2, NEF, ARW, DNG, etc.) |
| `opencv-python` | Accurate sharpness / noise / DoF metrics |
| `pillow-heif` | HEIC / HEIF / AVIF support |

Without these, the scripts fall back to PIL-only metrics and skip unsupported formats.

---

## photo_organizer.py

Scans a source folder, extracts EXIF dates, and copies files into a `YYYY/MM/` tree.
Detects RAW+JPG pairs, exact duplicates (by hash), and perceptual duplicates (pHash).

### Quick start

```bash
# Preview — no files written
python photo_organizer.py \
  --source /path/to/photos \
  --output /path/to/output \
  --deduplicate \
  --dry-run

# Run for real
python photo_organizer.py \
  --source /path/to/photos \
  --output /path/to/output \
  --deduplicate

# Full dedup: exact + perceptual, 8 hashing threads
python photo_organizer.py \
  --source /path/to/photos \
  --output /path/to/output \
  --deduplicate \
  --perceptual \
  --workers 8

# Use mtime when EXIF is absent; resume an interrupted run
python photo_organizer.py \
  --source /path/to/photos \
  --output /path/to/output \
  --fallback-to-mtime \
  --resume
```

### Output structure

```
/output/
  2023/08/               ← files with EXIF date
  2022/11/
  unknown/               ← files with no usable date
  _duplicates_review/    ← duplicates (never deleted, review manually)
  log.csv                ← full audit trail
  photo_organizer.log
  .photo_organizer_state.db   ← created only with --resume
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--source DIR` | required | Source directory (recursive) |
| `--output DIR` | required | Output root |
| `--deduplicate` | off | Detect and segregate exact duplicates |
| `--perceptual` | off | Also detect visually similar images via pHash |
| `--dry-run` | off | Preview actions without writing anything |
| `--fallback-to-mtime` | off | Use file mtime when EXIF date is absent |
| `--keep-raw-jpg-pairs` | off | Copy both RAW and JPG; default keeps RAW, reviews JPG |
| `--hash-algorithm` | `sha256` | `md5` or `sha256` |
| `--phash-threshold N` | `10` | Max Hamming distance for perceptual match |
| `--batch-size N` | `5000` | Files per exiftool batch call |
| `--workers N` | `4` | Threads for parallel hashing |
| `--resume` | off | Skip files already processed in a previous run |
| `--verbose` | off | Debug-level console output |

### Duplicate handling

Priority order (first match wins):

1. **RAW+JPG pairs** — same basename, mixed extensions (`IMG_1234.CR2` + `IMG_1234.JPG`). By default, RAW stays in the date tree; JPG goes to `_duplicates_review/`. Override with `--keep-raw-jpg-pairs`.
2. **Exact duplicates** — files grouped by size first (unique-size files skipped, ~80-90% I/O saved), then hashed. Best candidate (RAW > resolution > size) is kept.
3. **Perceptual duplicates** — pHash comparison scoped to same year/month to limit O(n²) cost.

### Performance

| Archive size | Metadata scan | Notes |
|---|---|---|
| 10k files | ~15 s | Batch exiftool, SSD |
| 100k files | ~2.5 min | Single exiftool call per 5000 files |
| 1M+ files | ~25 min | Hashing is the bottleneck |

---

## select_best_shots.py

Scores every image in a folder by composite quality metrics and copies the top picks.
Supports JPEG, PNG, TIFF, HEIC, WEBP, and RAW formats (CR2, NEF, ARW, DNG, and more — requires `rawpy`).

### Score components

| Metric | Role |
|---|---|
| Sharpness | Local Laplacian variance; strongest weight |
| Depth of field separation | Sharp vs. soft region contrast |
| Exposure | Distance from neutral brightness |
| Contrast | Dynamic range of the frame |
| Saturation | Colour richness |
| Lighting mood | Atmospheric / moody lighting reward |
| Noise penalty | High-ISO grain suppression |
| Clutter penalty | Busy backgrounds penalized |
| Highlight / oversat penalty | Clipped whites and oversaturation |
| ISO penalty | Progressive penalty above ISO 800 |

### Quick start

```bash
# Dry-run — see what would be selected
python select_best_shots.py \
  --source /path/to/photos \
  --output /path/to/best_shots \
  --top-percent 20 \
  --dry-run

# Select top 20%, 4 workers
python select_best_shots.py \
  --source /path/to/photos \
  --output /path/to/best_shots \
  --top-percent 20 \
  --workers 4

# Process RAW-only archive (requires rawpy)
python select_best_shots.py \
  --source /path/to/raw_photos \
  --output /path/to/best_shots \
  --top-percent 15 \
  --workers 4

# One folder per shooting day, each gets its own _best subfolder
python select_best_shots.py \
  --source /path/to/photos \
  --output /path/to/output \
  --split-by-subfolder \
  --top-percent 20
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--source DIR` | required | Input folder |
| `--output DIR` | required | Output folder |
| `--top-n N` | — | Keep exactly N files |
| `--top-percent P` | `40` | Keep top P% of scored files |
| `--workers N` | `1` | Parallel scoring processes |
| `--copy-mode` | `flat` | `flat` or `preserve-tree` |
| `--with-raw-pairs` | on | Copy RAW companion when selecting a JPEG |
| `--best-practice` | on | Technical floor + soft dedup pipeline |
| `--min-sharpness` | `0.20` | Floor threshold (0–1) |
| `--min-exposure` | `0.18` | Floor threshold (0–1) |
| `--min-contrast` | `0.12` | Floor threshold (0–1) |
| `--phash-threshold N` | `6` | Hamming distance for similar-shot grouping |
| `--low-key-friendly` | on | Reduce penalty for intentionally dark frames |
| `--prefer-low-key` | on | Bonus for dark frames with strong contrast |
| `--creative-dark-rescue` | on | Extra bonus for dark but structurally sharp frames |
| `--scoring-max-side N` | `1600` | Downscale long side before metric computation |
| `--split-by-subfolder` | off | Process each subfolder independently |
| `--dry-run` | off | Preview without copying |
| `--report PATH` | — | Write full TSV score table |

> **Note:** `--similarity-threshold` (float fraction) is deprecated. Use `--phash-threshold` (Hamming distance 1–64) instead.

---

## Requirements

- Python 3.11+
- `exiftool` system binary (photo_organizer.py only)
- See `requirements.txt` and `requirements-full.txt`
