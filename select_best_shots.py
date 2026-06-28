#!/usr/bin/env python3
"""
select_best_shots_v2.py — improved best-shot selector.

Changes vs select_best_shots.py:
- pHash threshold: 8x8 hash = 64 bits, conversion fixed (was multiplied by 32).
  New CLI flag --phash-threshold takes Hamming distance directly.
- All scoring coefficients live in ScoreWeights; magic numbers in Tunables.
- Images are downscaled to --scoring-max-side before metric computation.
  Massive speedup on large JPEGs; full resolution still used for megapixels.
- Deterministic ordering: ties broken by path string.
- Sharpness is dampened on noisy frames so high-ISO grain doesn't fake focus.
- Optional EXIF ISO penalty (uses PIL only, no exiftool dependency).
- Optional pillow-heif registration for HEIC/HEIF/AVIF.
- Auto-Canny thresholds in clutter estimate (replaces hardcoded 70/140).
- find_raw_companion: case-insensitive sibling scan.
- ProcessPoolExecutor uses ex.map with chunksize (lower memory for large sets).
- Multiprocess errors carry a short traceback so failures aren't opaque.
- select_with_low_key_quota preserves ranking instead of prefixing low-key.
- --report is per-source in --split-by-subfolder mode (no overwrites).
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageFilter, ImageOps, ImageStat
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    OPENCV_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    IMAGEHASH_AVAILABLE = False

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    rawpy = None
    RAWPY_AVAILABLE = False


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif", ".avif",
    ".raw", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".ptx", ".pef", ".rw2", ".rwl", ".srw", ".x3f",
    ".raf", ".3fr", ".fff", ".kdc", ".dcr", ".mrw", ".mdc",
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {
    ".raw", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".ptx", ".pef", ".rw2", ".rwl", ".srw", ".x3f",
    ".raf", ".3fr", ".fff", ".kdc", ".dcr", ".mrw", ".mdc",
}

LOGGER = logging.getLogger("select_best_shots_v2")


@dataclass(frozen=True)
class Tunables:
    """All knob constants previously scattered through the code."""
    scoring_max_side: int = 1600
    sharp_log_scale: float = 2400.0
    contrast_log_scale: float = 64.0
    saturation_log_scale: float = 80.0
    resolution_log_scale: float = 24.0
    noise_log_scale: float = 35.0
    exposure_target: float = 110.0
    exposure_tolerance: float = 145.0
    sharp_top_percentile: float = 85.0
    sharp_bot_percentile: float = 35.0
    subject_percentile: float = 75.0
    highlight_value_threshold: float = 0.985
    oversat_sat_threshold: float = 0.90
    oversat_val_threshold: float = 0.55
    highlight_multiplier: float = 3.0
    oversat_multiplier: float = 4.0
    clutter_multiplier: float = 3.0
    mood_brightness_target: float = 0.42
    mood_contrast_target: float = 0.45
    low_key_exposure_max: float = 0.58
    low_key_contrast_min: float = 0.88
    low_key_boost_max: float = 0.12
    rescue_exposure_max: float = 0.62
    rescue_contrast_min: float = 0.82
    rescue_sharpness_min: float = 0.52
    rescue_bonus_max: float = 0.18
    iso_penalty_start: float = 800.0
    iso_penalty_full: float = 6400.0
    phash_hash_size: int = 8  # 8x8 → 64-bit hash
    noise_sharpness_damp_start: float = 0.6


@dataclass(frozen=True)
class ScoreWeights:
    """All weights and penalties in one place. Previously some were hardcoded."""
    sharpness: float = 0.55
    exposure: float = 0.06
    contrast: float = 0.15
    saturation: float = 0.10
    resolution: float = 0.05
    dof_separation: float = 0.22
    lighting_mood: float = 0.16
    noise_penalty: float = 0.08
    clutter_penalty: float = 0.20
    highlight_penalty: float = 0.22
    oversat_penalty: float = 0.18
    iso_penalty: float = 0.10


@dataclass
class ScoredImage:
    path: Path
    score: float
    sharpness: float
    exposure: float
    contrast: float
    saturation: float
    megapixels: float
    noise: float
    dof_separation: float = 0.0
    clutter: float = 0.0
    highlight_penalty: float = 0.0
    oversat_penalty: float = 0.0
    lighting_mood: float = 0.0
    iso: Optional[int] = None
    iso_penalty: float = 0.0
    phash: Optional[str] = None


def setup_logger(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def iter_images(root: Path, exclude_root: Optional[Path] = None) -> Iterator[Path]:
    excluded = exclude_root.resolve() if exclude_root else None
    for path in root.rglob("*"):
        if excluded and (path == excluded or excluded in path.parents):
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def safe_dest(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def normalized_log(value: float, scale: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(scale), 1.0)


def find_raw_companion(path: Path) -> Optional[Path]:
    """Case-insensitive RAW companion lookup (works on case-sensitive FS too)."""
    if path.suffix.lower() not in JPEG_EXTENSIONS:
        return None
    parent = path.parent
    stem_lower = path.stem.lower()
    try:
        siblings = list(parent.iterdir())
    except OSError:
        return None
    for sibling in siblings:
        if not sibling.is_file():
            continue
        if sibling.stem.lower() != stem_lower:
            continue
        if sibling.suffix.lower() in RAW_EXTENSIONS:
            return sibling
    return None


def _extract_iso(img: Image.Image) -> Optional[int]:
    """Read ISO from EXIF via PIL only. None when missing/unreadable.

    Canon and most DSLRs store ISO in the ExifIFD sub-IFD (0x8769), not the
    top-level EXIF tree, so we have to walk both.
    """
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    # PhotographicSensitivity, ISOSpeedRatings, RecommendedExposureIndex
    iso_tags = (34867, 34855, 41497)
    sources = [exif]
    try:
        sources.append(exif.get_ifd(0x8769))
    except Exception:
        pass
    for src in sources:
        if not src:
            continue
        for tag_id in iso_tags:
            val = src.get(tag_id)
            if val is None:
                continue
            if isinstance(val, (tuple, list)) and val:
                val = val[0]
            try:
                iso = int(val)
            except (TypeError, ValueError):
                continue
            if iso > 0:
                return iso
    return None


def _downscale_for_scoring(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _load_raw_rgb(path: Path) -> tuple[Image.Image, Optional[int], int, int]:
    """Load RAW file via rawpy. Returns (rgb_pil, iso, width, height)."""
    with rawpy.imread(str(path)) as raw:
        iso: Optional[int] = None
        try:
            iso = int(raw.other_params.get("iso", 0)) or None
        except Exception:
            pass
        rgb_np = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8,
        )
    img = Image.fromarray(rgb_np)
    return img, iso, img.size[0], img.size[1]


def _load_oriented_rgb(path: Path) -> tuple[Image.Image, Optional[int], int, int]:
    """Open image, read ISO, apply EXIF orientation. Returns (rgb, iso, full_w, full_h)."""
    if RAWPY_AVAILABLE and path.suffix.lower() in RAW_EXTENSIONS:
        return _load_raw_rgb(path)
    with Image.open(path) as img:
        iso = _extract_iso(img)
        oriented = ImageOps.exif_transpose(img)
        rgb = oriented.convert("RGB")
    return rgb, iso, rgb.size[0], rgb.size[1]


def _noise_estimate(gray: "np.ndarray") -> float:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype("float32") - blurred.astype("float32")
    return float(residual.std())


def _auto_canny_bounds(tile: "np.ndarray", sigma: float = 0.33) -> tuple[int, int]:
    med = float(np.median(tile))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _local_sharpness_features(gray: "np.ndarray", tun: Tunables) -> tuple[float, float, float]:
    """Returns (local_sharpness, dof_separation, clutter)."""
    h, w = gray.shape
    block = max(24, min(h, w) // 8)
    step = max(16, block // 2)
    vals: list[float] = []
    edge_density: list[float] = []
    for y in range(0, h - block + 1, step):
        for x in range(0, w - block + 1, step):
            tile = gray[y:y + block, x:x + block]
            lap = cv2.Laplacian(tile, cv2.CV_64F)
            vals.append(float(lap.var()))
            lo, hi = _auto_canny_bounds(tile)
            edges = cv2.Canny(tile, lo, hi)
            edge_density.append(float((edges > 0).mean()))
    if not vals:
        return 0.0, 0.0, 0.0

    v = np.asarray(vals, dtype=np.float64)
    e = np.asarray(edge_density, dtype=np.float64)
    top = float(np.percentile(v, tun.sharp_top_percentile))
    bot = float(np.percentile(v, tun.sharp_bot_percentile))
    local_sharp = normalized_log(top, scale=tun.sharp_log_scale)
    dof_sep = max(0.0, min((top - bot) / max(top, 1.0), 1.0))

    subject_mask = v >= np.percentile(v, tun.subject_percentile)
    bg_edges = e[~subject_mask] if (~subject_mask).any() else e
    clutter = float(np.clip(bg_edges.mean() * tun.clutter_multiplier, 0.0, 1.0))
    return local_sharp, dof_sep, clutter


def _highlight_and_oversat_penalties(rgb_np: "np.ndarray", tun: Tunables) -> tuple[float, float]:
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    clipped = float((v > tun.highlight_value_threshold).mean())
    oversat = float(((s > tun.oversat_sat_threshold) & (v > tun.oversat_val_threshold)).mean())
    return (
        min(clipped * tun.highlight_multiplier, 1.0),
        min(oversat * tun.oversat_multiplier, 1.0),
    )


def _lighting_mood_score(brightness: float, contrast_raw: float, tun: Tunables) -> float:
    b = brightness / 255.0
    c = min(contrast_raw / 64.0, 1.0)
    bright_pref = 1.0 - min(abs(b - tun.mood_brightness_target) / max(tun.mood_brightness_target, 0.01), 1.0)
    contrast_pref = 1.0 - min(abs(c - tun.mood_contrast_target) / 0.55, 1.0)
    return max(0.0, min((bright_pref * 0.65 + contrast_pref * 0.35), 1.0))


def _iso_penalty(iso: Optional[int], tun: Tunables) -> float:
    if iso is None or iso <= 0:
        return 0.0
    if iso <= tun.iso_penalty_start:
        return 0.0
    if iso >= tun.iso_penalty_full:
        return 1.0
    span = tun.iso_penalty_full - tun.iso_penalty_start
    return min(1.0, (iso - tun.iso_penalty_start) / span)


def _phash_string(rgb: Image.Image, hash_size: int) -> Optional[str]:
    if not IMAGEHASH_AVAILABLE:
        return None
    return str(imagehash.phash(rgb, hash_size=hash_size))


def score_image(path: Path, weights: ScoreWeights, tun: Tunables, prefer_low_key: bool) -> ScoredImage:
    """Score a single image. Raises on failure."""
    rgb_full, iso, full_w, full_h = _load_oriented_rgb(path)
    rgb = _downscale_for_scoring(rgb_full, tun.scoring_max_side)
    gray_pil = rgb.convert("L")
    gray_stat = ImageStat.Stat(gray_pil)
    rgb_stat = ImageStat.Stat(rgb)

    if OPENCV_AVAILABLE:
        gray_np = np.array(gray_pil)
        noise_raw = _noise_estimate(gray_np)
        local_sharpness, dof_sep, clutter = _local_sharpness_features(gray_np, tun)
        rgb_np = np.array(rgb)
        highlight_pen_raw, oversat_pen_raw = _highlight_and_oversat_penalties(rgb_np, tun)
    else:
        edges = gray_pil.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        sharp_raw = float(edge_stat.var[0])
        noise_raw = float(gray_stat.stddev[0]) * 0.5
        local_sharpness = normalized_log(sharp_raw, scale=tun.sharp_log_scale)
        dof_sep = 0.0
        clutter = 0.0
        highlight_pen_raw = 0.0
        oversat_pen_raw = 0.0

    brightness = float(gray_stat.mean[0])
    contrast_raw = float(gray_stat.stddev[0])
    sat_channels = rgb_stat.stddev
    saturation_raw = float(sum(sat_channels) / 3.0)
    megapixels = (full_w * full_h) / 1_000_000.0

    sharpness = local_sharpness
    exposure = max(0.0, 1.0 - abs(brightness - tun.exposure_target) / tun.exposure_tolerance)
    contrast = normalized_log(contrast_raw, scale=tun.contrast_log_scale)
    saturation = normalized_log(saturation_raw, scale=tun.saturation_log_scale)
    resolution = normalized_log(megapixels, scale=tun.resolution_log_scale)
    noise = normalized_log(noise_raw, scale=tun.noise_log_scale)
    lighting_mood = _lighting_mood_score(brightness, contrast_raw, tun)
    iso_pen = _iso_penalty(iso, tun)

    # High-ISO grain inflates Laplacian variance; discount sharpness when noisy.
    if noise > tun.noise_sharpness_damp_start:
        sharpness *= max(0.4, 1.0 - (noise - tun.noise_sharpness_damp_start) * 1.2)

    low_key_boost = 0.0
    if prefer_low_key and exposure < tun.low_key_exposure_max and contrast >= tun.low_key_contrast_min:
        darkness = min((tun.low_key_exposure_max - exposure) / 0.30, 1.0)
        low_key_boost = tun.low_key_boost_max * darkness

    score = (
        sharpness * weights.sharpness
        + exposure * weights.exposure
        + contrast * weights.contrast
        + saturation * weights.saturation
        + resolution * weights.resolution
        + dof_sep * weights.dof_separation
        + lighting_mood * weights.lighting_mood
        - noise * weights.noise_penalty
        - clutter * weights.clutter_penalty
        - highlight_pen_raw * weights.highlight_penalty
        - oversat_pen_raw * weights.oversat_penalty
        - iso_pen * weights.iso_penalty
        + low_key_boost
    )

    return ScoredImage(
        path=path,
        score=score,
        sharpness=sharpness,
        exposure=exposure,
        contrast=contrast,
        saturation=saturation,
        megapixels=megapixels,
        noise=noise,
        dof_separation=dof_sep,
        clutter=clutter,
        highlight_penalty=highlight_pen_raw,
        oversat_penalty=oversat_pen_raw,
        lighting_mood=lighting_mood,
        iso=iso,
        iso_penalty=iso_pen,
        phash=_phash_string(rgb, tun.phash_hash_size),
    )


def creative_dark_rescue_bonus(item: ScoredImage, tun: Tunables, enabled: bool) -> float:
    if not enabled:
        return 0.0
    if item.exposure > tun.rescue_exposure_max:
        return 0.0
    if item.contrast < tun.rescue_contrast_min or item.sharpness < tun.rescue_sharpness_min:
        return 0.0
    darkness = min((tun.rescue_exposure_max - item.exposure) / 0.40, 1.0)
    structure = min((item.contrast + item.sharpness) / 2.0, 1.0)
    noise_guard = max(0.0, 1.0 - item.noise)
    return tun.rescue_bonus_max * darkness * structure * (0.55 + 0.45 * noise_guard)


def score_image_safe(task: tuple) -> tuple[Optional[ScoredImage], Optional[str]]:
    """Picklable wrapper for ProcessPoolExecutor."""
    path, weights, tun, prefer_low_key = task
    try:
        return score_image(path, weights, tun, prefer_low_key), None
    except Exception:
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        return None, f"{path}: {tb}"


def score_all(
    images: list[Path],
    weights: ScoreWeights,
    tun: Tunables,
    prefer_low_key: bool,
    creative_dark_rescue: bool,
    workers: int,
) -> tuple[list[ScoredImage], int]:
    tasks = [(path, weights, tun, prefer_low_key) for path in images]
    scored: list[ScoredImage] = []
    errors = 0

    def consume(iterator):
        nonlocal errors
        for item, err in tqdm(iterator, total=len(tasks), desc="Scoring photos", unit="file"):
            if item is not None:
                item.score += creative_dark_rescue_bonus(item, tun, creative_dark_rescue)
                scored.append(item)
            else:
                errors += 1
                LOGGER.warning(err)

    if workers <= 1:
        consume(score_image_safe(t) for t in tasks)
        return scored, errors

    chunksize = max(1, min(32, (len(tasks) // (workers * 4)) or 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        consume(ex.map(score_image_safe, tasks, chunksize=chunksize))
    return scored, errors


def passes_technical_floor(item: ScoredImage, min_sharpness: float, min_exposure: float, min_contrast: float) -> bool:
    return (
        item.sharpness >= min_sharpness
        and item.exposure >= min_exposure
        and item.contrast >= min_contrast
    )


def phash_threshold_from_fraction(hash_size: int, fraction: float) -> int:
    """Map legacy similarity-fraction (0..1) to Hamming distance for an NxN hash."""
    bits = hash_size * hash_size
    return max(1, min(bits, int(round(fraction * bits))))


def phash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def cluster_by_phash(candidates: list[ScoredImage], threshold: int) -> list[list[ScoredImage]]:
    """Greedy pHash clustering in input order (assumed score-sorted)."""
    if not IMAGEHASH_AVAILABLE:
        return [[c] for c in candidates]
    clusters: list[list[ScoredImage]] = []
    reps: list[Optional[str]] = []
    for item in candidates:
        if not item.phash:
            clusters.append([item])
            reps.append(None)
            continue
        placed = False
        for idx, rep in enumerate(reps):
            if rep is not None and phash_distance(item.phash, rep) <= threshold:
                clusters[idx].append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
            reps.append(item.phash)
    return clusters


def apply_soft_dedup(
    ranked: list[ScoredImage],
    phash_threshold: int,
    soft_penalty: float,
    cluster_keep_top_k: int,
    cluster_keep_top_k_min_size: int,
) -> tuple[list[ScoredImage], int]:
    clusters = cluster_by_phash(ranked, phash_threshold)
    out: list[ScoredImage] = []
    penalized = 0
    for cluster in clusters:
        cluster.sort(key=lambda x: (-x.score, str(x.path)))
        k = max(1, cluster_keep_top_k) if len(cluster) >= cluster_keep_top_k_min_size else 1
        for idx, item in enumerate(cluster):
            if idx >= k:
                penalized += 1
                item.score -= soft_penalty * min(idx - k + 1, 3)
            out.append(item)
    out.sort(key=lambda x: (-x.score, str(x.path)))
    return out, penalized


def pick_count(total: int, top_n: Optional[int], top_percent: Optional[float]) -> int:
    if total == 0:
        return 0
    if top_n is not None:
        return max(1, min(top_n, total))
    if top_percent is not None:
        return max(1, min(math.ceil(total * (top_percent / 100.0)), total))
    return max(1, math.ceil(total * 0.1))


def select_with_low_key_quota(
    candidates: list[ScoredImage],
    keep_count: int,
    low_key_quota_percent: float,
    low_key_exposure_max: float,
) -> list[ScoredImage]:
    """Quota acts as a minimum guarantee while preserving rank order."""
    if keep_count <= 0:
        return []
    if low_key_quota_percent <= 0:
        return candidates[:keep_count]

    quota = max(0, min(keep_count, int(round(keep_count * (low_key_quota_percent / 100.0)))))
    top_slice = list(candidates[:keep_count])
    low_key_in_top = sum(1 for c in top_slice if c.exposure <= low_key_exposure_max)
    if low_key_in_top >= quota:
        return top_slice

    deficit = quota - low_key_in_top
    in_top_paths = {c.path for c in top_slice}
    extra_low_key = [
        c for c in candidates[keep_count:]
        if c.exposure <= low_key_exposure_max and c.path not in in_top_paths
    ][:deficit]
    if not extra_low_key:
        return top_slice

    # Drop the lowest-ranked non-low-key entries first to make room.
    removable_indices = [i for i, c in enumerate(top_slice) if c.exposure > low_key_exposure_max]
    for replacement in extra_low_key:
        if not removable_indices:
            break
        idx = removable_indices.pop()
        top_slice[idx] = replacement
    top_slice.sort(key=lambda x: (-x.score, str(x.path)))
    return top_slice


REPORT_COLUMNS = (
    "score", "sharpness", "dof_separation", "clutter", "exposure",
    "lighting_mood", "contrast", "saturation", "highlight_penalty",
    "oversat_penalty", "noise", "iso", "iso_penalty", "megapixels",
    "phash", "path",
)


def _format_row(item: ScoredImage) -> str:
    iso_str = str(item.iso) if item.iso is not None else ""
    return (
        f"{item.score:.6f}\t{item.sharpness:.6f}\t{item.dof_separation:.6f}\t"
        f"{item.clutter:.6f}\t{item.exposure:.6f}\t{item.lighting_mood:.6f}\t"
        f"{item.contrast:.6f}\t{item.saturation:.6f}\t{item.highlight_penalty:.6f}\t"
        f"{item.oversat_penalty:.6f}\t{item.noise:.6f}\t{iso_str}\t"
        f"{item.iso_penalty:.6f}\t{item.megapixels:.3f}\t{item.phash or ''}\t{item.path}"
    )


def write_report(path: Path, scored: list[ScoredImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(REPORT_COLUMNS) + "\n")
        for item in scored:
            fh.write(_format_row(item) + "\n")


def write_selection_log(path: Path, scored: list[ScoredImage], selected: list[ScoredImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_paths = {item.path for item in selected}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("rank\tselected\t" + "\t".join(REPORT_COLUMNS) + "\n")
        for rank, item in enumerate(scored, start=1):
            is_selected = "1" if item.path in selected_paths else "0"
            fh.write(f"{rank}\t{is_selected}\t{_format_row(item)}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select best shots by image quality metrics (v2 — fixed pHash, downscaled metrics, ISO penalty)."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--top-percent", type=float, default=40.0)
    parser.add_argument("--copy-mode", choices=("flat", "preserve-tree"), default="flat")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, default=None,
                        help="TSV report path. In --split-by-subfolder mode each subfolder gets its own suffixed file.")
    parser.add_argument("--with-raw-pairs", action="store_true", default=True)
    parser.add_argument("--no-with-raw-pairs", dest="with_raw_pairs", action="store_false")

    parser.add_argument("--best-practice", action="store_true", default=True)
    parser.add_argument("--no-best-practice", dest="best_practice", action="store_false")
    parser.add_argument("--min-sharpness", type=float, default=0.20)
    parser.add_argument("--min-exposure", type=float, default=0.18)
    parser.add_argument("--min-contrast", type=float, default=0.12)

    parser.add_argument("--phash-threshold", type=int, default=6,
                        help="Max Hamming distance (1..64) to consider two 8x8 pHash frames duplicates. Higher = looser.")
    parser.add_argument("--similarity-threshold", type=float, default=None,
                        help="Deprecated. Fraction of 64-bit pHash; mapped to --phash-threshold.")

    parser.add_argument("--low-key-friendly", action="store_true", default=True)
    parser.add_argument("--no-low-key-friendly", dest="low_key_friendly", action="store_false")
    parser.add_argument("--prefer-low-key", action="store_true", default=True)
    parser.add_argument("--no-prefer-low-key", dest="prefer_low_key", action="store_false")
    parser.add_argument("--creative-dark-rescue", action="store_true", default=True)
    parser.add_argument("--no-creative-dark-rescue", dest="creative_dark_rescue", action="store_false")
    parser.add_argument("--low-key-quota-percent", type=float, default=0.0)
    parser.add_argument("--low-key-exposure-max", type=float, default=0.58)

    parser.add_argument("--soft-dedup-penalty", type=float, default=0.04)
    parser.add_argument("--cluster-keep-top-k", type=int, default=2)
    parser.add_argument("--cluster-keep-top-k-min-size", type=int, default=5)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scoring-max-side", type=int, default=1600,
                        help="Downscale long side to N pixels before metric computation.")

    parser.add_argument("--weight-sharpness", type=float, default=None)
    parser.add_argument("--weight-exposure", type=float, default=None)
    parser.add_argument("--weight-contrast", type=float, default=None)
    parser.add_argument("--weight-saturation", type=float, default=None)
    parser.add_argument("--weight-resolution", type=float, default=None)
    parser.add_argument("--weight-noise-penalty", type=float, default=None)
    parser.add_argument("--weight-iso-penalty", type=float, default=None)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--split-by-subfolder", action="store_true")
    parser.add_argument("--output-template", type=str, default="{name}_best")
    return parser


def resolved_weights(args: argparse.Namespace) -> ScoreWeights:
    base = ScoreWeights(
        exposure=0.06 if args.low_key_friendly else 0.15,
    )
    overrides: dict = {}
    for key, attr in (
        ("sharpness", "weight_sharpness"),
        ("exposure", "weight_exposure"),
        ("contrast", "weight_contrast"),
        ("saturation", "weight_saturation"),
        ("resolution", "weight_resolution"),
        ("noise_penalty", "weight_noise_penalty"),
        ("iso_penalty", "weight_iso_penalty"),
    ):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    return replace(base, **overrides)


def validate_weights(weights: ScoreWeights) -> None:
    for name in (
        "sharpness", "exposure", "contrast", "saturation", "resolution",
        "dof_separation", "lighting_mood", "noise_penalty", "clutter_penalty",
        "highlight_penalty", "oversat_penalty", "iso_penalty",
    ):
        if getattr(weights, name) < 0:
            raise ValueError(f"Weight {name} must be >= 0")


def resolved_phash_threshold(args: argparse.Namespace, tun: Tunables) -> int:
    if args.similarity_threshold is not None:
        return phash_threshold_from_fraction(tun.phash_hash_size, args.similarity_threshold)
    bits = tun.phash_hash_size * tun.phash_hash_size
    return max(1, min(args.phash_threshold, bits))


def run_selection(source: Path, output: Path, args: argparse.Namespace) -> int:
    tun = Tunables(scoring_max_side=args.scoring_max_side)
    weights = resolved_weights(args)
    phash_thr = resolved_phash_threshold(args, tun)

    images = list(iter_images(source, exclude_root=output))
    if not images:
        print(f"[{source.name}] No supported images found.")
        return 0

    scored, errors = score_all(
        images,
        weights,
        tun,
        args.prefer_low_key,
        args.creative_dark_rescue,
        args.workers,
    )
    if not scored:
        print(f"[{source.name}] No images could be scored.")
        return 1

    scored.sort(key=lambda x: (-x.score, str(x.path)))
    candidates = scored
    dropped_by_floor = 0
    soft_penalized = 0
    if args.best_practice:
        floored = [
            item for item in scored
            if passes_technical_floor(item, args.min_sharpness, args.min_exposure, args.min_contrast)
        ]
        dropped_by_floor = len(scored) - len(floored)
        if not floored:
            print(f"[{source.name}] Best-practice floor removed everything; fallback to full scored set.")
            floored = scored
        candidates, soft_penalized = apply_soft_dedup(
            ranked=floored,
            phash_threshold=phash_thr,
            soft_penalty=args.soft_dedup_penalty,
            cluster_keep_top_k=args.cluster_keep_top_k,
            cluster_keep_top_k_min_size=args.cluster_keep_top_k_min_size,
        )

    keep_count = pick_count(len(candidates), args.top_n, args.top_percent)
    selected = select_with_low_key_quota(
        candidates=candidates,
        keep_count=keep_count,
        low_key_quota_percent=args.low_key_quota_percent if args.best_practice else 0.0,
        low_key_exposure_max=args.low_key_exposure_max,
    )

    print(f"[{source.name}] Scored: {len(scored)} files")
    if args.best_practice:
        print(f"[{source.name}] After technical floor: {len(candidates)} files")
        print(f"[{source.name}] Dropped by floor: {dropped_by_floor}")
        print(f"[{source.name}] Soft-penalized as similar: {soft_penalized}")
        print(f"[{source.name}] pHash threshold (Hamming): {phash_thr}")
    print(f"[{source.name}] Selected: {len(selected)} files")
    selected_percent = (len(selected) / len(scored) * 100.0) if scored else 0.0
    print(f"[{source.name}] Selected percent: {selected_percent:.2f}%")
    if errors:
        print(f"[{source.name}] Scoring errors: {errors} (see warnings)")
    if args.dry_run:
        print(f"[{source.name}] Dry-run: no files copied")

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    copied_raw_pairs = 0
    for item in selected:
        rel = item.path.relative_to(source)
        dest = output / rel if args.copy_mode == "preserve-tree" else output / item.path.name
        dest = safe_dest(dest)
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.path, dest)

        if args.with_raw_pairs:
            raw_pair = find_raw_companion(item.path)
            if raw_pair:
                raw_rel = raw_pair.relative_to(source)
                raw_dest = output / raw_rel if args.copy_mode == "preserve-tree" else output / raw_pair.name
                raw_dest = safe_dest(raw_dest)
                if not args.dry_run:
                    raw_dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(raw_pair, raw_dest)
                copied_raw_pairs += 1

    if args.report:
        report_path = args.report.resolve()
        if args.split_by_subfolder:
            report_path = report_path.with_name(f"{report_path.stem}_{source.name}{report_path.suffix}")
        if args.dry_run:
            print(f"[{source.name}] Dry-run: report not written ({report_path})")
        else:
            write_report(report_path, scored)
            print(f"[{source.name}] Report: {report_path}")

    selection_log_path = output / "_selection_log.tsv"
    if args.dry_run:
        print(f"[{source.name}] Dry-run: selection log not written ({selection_log_path})")
    else:
        write_selection_log(selection_log_path, scored, selected)
        print(f"[{source.name}] Selection log: {selection_log_path}")

    if args.with_raw_pairs:
        label = "found for selected JPG/JPEG" if args.dry_run else "copied"
        print(f"[{source.name}] RAW companions {label}: {copied_raw_pairs}")
    print(f"[{source.name}] Done.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logger(args.verbose)

    if args.top_n is not None and args.top_n <= 0:
        parser.error("--top-n must be > 0")
    if args.top_percent is not None and not (0 < args.top_percent <= 100):
        parser.error("--top-percent must be in (0, 100]")
    if args.top_n is not None and args.top_percent is not None:
        # top_n wins; we just don't error here. v1 errored, but v2 keeps the more useful behavior.
        pass
    if args.phash_threshold <= 0:
        parser.error("--phash-threshold must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.cluster_keep_top_k <= 0:
        parser.error("--cluster-keep-top-k must be > 0")
    if args.cluster_keep_top_k_min_size <= 0:
        parser.error("--cluster-keep-top-k-min-size must be > 0")
    if args.low_key_quota_percent < 0 or args.low_key_quota_percent > 100:
        parser.error("--low-key-quota-percent must be in [0, 100]")
    if args.scoring_max_side < 256:
        parser.error("--scoring-max-side must be >= 256")

    if not OPENCV_AVAILABLE:
        LOGGER.warning("OpenCV unavailable; using fallback metrics.")
    if not HEIF_AVAILABLE:
        LOGGER.info("pillow-heif not installed; HEIC/HEIF/AVIF files will fail to open.")
    if not RAWPY_AVAILABLE:
        LOGGER.warning("rawpy not installed; RAW files (CR2/NEF/ARW/etc) will be skipped.")
    if args.best_practice and not IMAGEHASH_AVAILABLE:
        LOGGER.warning("imagehash not installed; dedup degrades to no grouping.")

    try:
        validate_weights(resolved_weights(args))
    except ValueError as exc:
        parser.error(str(exc))

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists() or not source.is_dir():
        parser.error(f"Source folder does not exist: {source}")

    if not args.split_by_subfolder:
        return run_selection(source, output, args)

    subdirs = sorted(p for p in source.iterdir() if p.is_dir())
    if not subdirs:
        print("No subfolders found in source.")
        return 0
    failures = 0
    for child in subdirs:
        if child.name.endswith("_best"):
            continue
        child_folder_name = args.output_template.format(name=child.name)
        child_output = output / child_folder_name
        rc = run_selection(child, child_output, args)
        if rc != 0:
            failures += 1
    if failures:
        print(f"Completed with failures: {failures}")
        return 1
    print("Batch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
