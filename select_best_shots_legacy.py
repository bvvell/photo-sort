#!/usr/bin/env python3
# DEPRECATED — use select_best_shots.py instead.
# This file is kept for reference only and will be removed in a future release.
"""
select_best_shots.py — pick the sharpest/most technically pleasing photos.

Scans a folder recursively, scores each image, and copies top-ranked shots
to a separate output folder.
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image, ImageFilter, ImageOps, ImageStat
from tqdm import tqdm

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


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic", ".heif", ".avif"
}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
RAW_EXTENSIONS = {
    ".raw", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
    ".dng", ".orf", ".ptx", ".pef", ".rw2", ".rwl", ".srw", ".x3f",
    ".raf", ".3fr", ".fff", ".kdc", ".dcr", ".mrw", ".mdc",
}


@dataclass(frozen=True)
class ScoreWeights:
    """Configurable score weights."""
    sharpness: float = 0.42
    exposure: float = 0.14
    contrast: float = 0.08
    saturation: float = 0.08
    resolution: float = 0.05
    noise_penalty: float = 0.08


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
    phash: Optional[str] = None


LOGGER = logging.getLogger("select_best_shots")

# Default curation profile (atmospheric / shallow-DOF friendly).
DEFAULT_BEST_PRACTICE = True
DEFAULT_SIMILARITY_THRESHOLD = 0.03
DEFAULT_TOP_PERCENT = 40.0
DEFAULT_LOW_KEY_FRIENDLY = True
DEFAULT_PREFER_LOW_KEY = True
DEFAULT_CREATIVE_DARK_RESCUE = True


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
    if path.suffix.lower() not in JPEG_EXTENSIONS:
        return None
    for ext in RAW_EXTENSIONS:
        candidate = path.with_suffix(ext)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def resolved_weights(args: argparse.Namespace) -> ScoreWeights:
    exposure_default = 0.06 if args.low_key_friendly else 0.15
    return ScoreWeights(
        sharpness=args.weight_sharpness if args.weight_sharpness is not None else 0.55,
        exposure=args.weight_exposure if args.weight_exposure is not None else exposure_default,
        contrast=args.weight_contrast if args.weight_contrast is not None else 0.15,
        saturation=args.weight_saturation if args.weight_saturation is not None else 0.10,
        resolution=args.weight_resolution if args.weight_resolution is not None else 0.05,
        noise_penalty=args.weight_noise_penalty if args.weight_noise_penalty is not None else 0.08,
    )


def validate_weights(weights: ScoreWeights) -> None:
    vals = (
        weights.sharpness,
        weights.exposure,
        weights.contrast,
        weights.saturation,
        weights.resolution,
        weights.noise_penalty,
    )
    if any(v < 0 for v in vals):
        raise ValueError("All weights must be >= 0")


def _load_oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img)
        return oriented.convert("RGB")


def _laplacian_variance(gray: "np.ndarray") -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _noise_estimate(gray: "np.ndarray") -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype("float32") - blurred.astype("float32")
    return float(residual.std())


def _local_sharpness_features(gray: "np.ndarray") -> tuple[float, float, float]:
    """
    Returns (local_sharpness, dof_separation, clutter).
    local_sharpness: sharpest 20% regions average.
    dof_separation: sharp regions minus soft regions.
    clutter: edge density in non-subject regions.
    """
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
            edges = cv2.Canny(tile, 70, 140)
            edge_density.append(float((edges > 0).mean()))
    if not vals:
        return 0.0, 0.0, 0.0

    v = np.array(vals, dtype=np.float64)
    e = np.array(edge_density, dtype=np.float64)
    top = float(np.percentile(v, 85))
    bot = float(np.percentile(v, 35))
    local_sharp = normalized_log(top, scale=2400.0)
    dof_sep = max(0.0, min((top - bot) / max(top, 1.0), 1.0))

    subject_mask = v >= np.percentile(v, 75)
    bg_edges = e[~subject_mask] if (~subject_mask).any() else e
    clutter = float(np.clip(bg_edges.mean() * 3.0, 0.0, 1.0))
    return local_sharp, dof_sep, clutter


def _highlight_and_oversat_penalties(rgb_np: "np.ndarray") -> tuple[float, float]:
    """Penalize clipped highlights and aggressive oversaturation."""
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    clipped = float((v > 0.985).mean())
    oversat = float(((s > 0.90) & (v > 0.55)).mean())
    return min(clipped * 3.0, 1.0), min(oversat * 4.0, 1.0)


def _lighting_mood_score(brightness: float, contrast_raw: float) -> float:
    """
    Reward soft/moody lighting:
    - brightness can be lower than neutral
    - moderate (not extreme) contrast is acceptable
    """
    b = brightness / 255.0
    c = min(contrast_raw / 64.0, 1.0)
    bright_pref = 1.0 - min(abs(b - 0.42) / 0.42, 1.0)
    contrast_pref = 1.0 - min(abs(c - 0.45) / 0.55, 1.0)
    return max(0.0, min((bright_pref * 0.65 + contrast_pref * 0.35), 1.0))


def _phash_string(rgb: Image.Image) -> Optional[str]:
    if not IMAGEHASH_AVAILABLE:
        return None
    return str(imagehash.phash(rgb))


def score_image(path: Path, weights: ScoreWeights, prefer_low_key: bool) -> ScoredImage:
    """Score a single image; raises on failure."""
    rgb = _load_oriented_rgb(path)
    gray_pil = rgb.convert("L")
    gray_stat = ImageStat.Stat(gray_pil)
    rgb_stat = ImageStat.Stat(rgb)

    if OPENCV_AVAILABLE:
        gray_np = np.array(gray_pil)
        sharp_raw = _laplacian_variance(gray_np)
        noise_raw = _noise_estimate(gray_np)
        local_sharpness, dof_sep, clutter = _local_sharpness_features(gray_np)
        rgb_np = np.array(rgb)
        highlight_penalty, oversat_penalty = _highlight_and_oversat_penalties(rgb_np)
    else:
        # Backward-compatible fallback when cv2 is unavailable.
        edges = gray_pil.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        sharp_raw = float(edge_stat.var[0])
        noise_raw = float(gray_stat.stddev[0]) * 0.5
        local_sharpness = normalized_log(sharp_raw, scale=2400.0)
        dof_sep = 0.0
        clutter = 0.0
        highlight_penalty = 0.0
        oversat_penalty = 0.0

    brightness = float(gray_stat.mean[0])
    contrast_raw = float(gray_stat.stddev[0])
    sat_channels = rgb_stat.stddev
    saturation_raw = float(sum(sat_channels) / 3.0)

    width, height = rgb.size
    megapixels = (width * height) / 1_000_000.0

    sharpness = local_sharpness
    exposure = max(0.0, 1.0 - abs(brightness - 110.0) / 145.0)
    contrast = normalized_log(contrast_raw, scale=64.0)
    saturation = normalized_log(saturation_raw, scale=80.0)
    resolution = normalized_log(megapixels, scale=24.0)
    noise = normalized_log(noise_raw, scale=35.0)
    lighting_mood = _lighting_mood_score(brightness, contrast_raw)

    low_key_boost = 0.0
    if prefer_low_key and exposure < 0.58 and contrast >= 0.88:
        darkness = min((0.58 - exposure) / 0.30, 1.0)
        low_key_boost = 0.12 * darkness

    score = (
        sharpness * weights.sharpness
        + exposure * weights.exposure
        + contrast * weights.contrast
        + saturation * weights.saturation
        + resolution * weights.resolution
        + dof_sep * 0.22
        + lighting_mood * 0.16
        - noise * weights.noise_penalty
        - clutter * 0.20
        - highlight_penalty * 0.22
        - oversat_penalty * 0.18
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
        highlight_penalty=highlight_penalty,
        oversat_penalty=oversat_penalty,
        lighting_mood=lighting_mood,
        phash=_phash_string(rgb),
    )


def creative_dark_rescue_bonus(item: ScoredImage, enabled: bool) -> float:
    """
    Rescue intentionally dark frames when structure is still strong.
    """
    if not enabled:
        return 0.0
    if item.exposure > 0.62:
        return 0.0
    if item.contrast < 0.82 or item.sharpness < 0.52:
        return 0.0
    darkness = min((0.62 - item.exposure) / 0.40, 1.0)
    structure = min((item.contrast + item.sharpness) / 2.0, 1.0)
    noise_guard = max(0.0, 1.0 - item.noise)
    return 0.18 * darkness * structure * (0.55 + 0.45 * noise_guard)


def score_image_safe(path: Path, weights: ScoreWeights, prefer_low_key: bool) -> tuple[Optional[ScoredImage], Optional[str]]:
    """Type-safe wrapper for multiprocessing."""
    try:
        return score_image(path, weights, prefer_low_key), None
    except Exception as exc:
        return None, f"{path}: {exc}"


def score_all(
    images: list[Path],
    weights: ScoreWeights,
    prefer_low_key: bool,
    creative_dark_rescue: bool,
    workers: int,
) -> tuple[list[ScoredImage], int]:
    scored: list[ScoredImage] = []
    errors = 0
    if workers <= 1:
        for path in tqdm(images, desc="Scoring photos", unit="file"):
            item, err = score_image_safe(path, weights, prefer_low_key)
            if item is not None:
                item.score += creative_dark_rescue_bonus(item, creative_dark_rescue)
                scored.append(item)
            else:
                errors += 1
                LOGGER.warning(err)
        return scored, errors

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(score_image_safe, path, weights, prefer_low_key): path
            for path in images
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring photos", unit="file"):
            item, err = fut.result()
            if item is not None:
                item.score += creative_dark_rescue_bonus(item, creative_dark_rescue)
                scored.append(item)
            else:
                errors += 1
                LOGGER.warning(err)
    return scored, errors


def passes_technical_floor(item: ScoredImage, min_sharpness: float, min_exposure: float, min_contrast: float) -> bool:
    return (
        item.sharpness >= min_sharpness
        and item.exposure >= min_exposure
        and item.contrast >= min_contrast
    )


def similarity_to_phash_threshold(similarity_threshold: float) -> int:
    """Map legacy float threshold to pHash Hamming distance."""
    return max(1, min(16, int(round(similarity_threshold * 32))))


def phash_distance(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def cluster_by_phash(candidates: list[ScoredImage], threshold: int) -> list[list[ScoredImage]]:
    """Greedy pHash clustering in score order."""
    if not IMAGEHASH_AVAILABLE:
        return [[c] for c in candidates]
    clusters: list[list[ScoredImage]] = []
    reps: list[str] = []
    for item in candidates:
        if not item.phash:
            clusters.append([item])
            reps.append("")
            continue
        placed = False
        for idx, rep in enumerate(reps):
            if rep and phash_distance(item.phash, rep) <= threshold:
                clusters[idx].append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
            reps.append(item.phash)
    return clusters


def apply_soft_dedup(
    ranked: list[ScoredImage],
    similarity_threshold: float,
    soft_penalty: float,
    cluster_keep_top_k: int,
    cluster_keep_top_k_min_size: int,
) -> tuple[list[ScoredImage], int]:
    """Apply soft dedup via pHash clusters."""
    threshold = similarity_to_phash_threshold(similarity_threshold)
    clusters = cluster_by_phash(ranked, threshold)
    out: list[ScoredImage] = []
    penalized = 0
    for cluster in clusters:
        cluster.sort(key=lambda x: x.score, reverse=True)
        k = 1
        if len(cluster) >= cluster_keep_top_k_min_size:
            k = max(1, cluster_keep_top_k)
        for idx, item in enumerate(cluster):
            if idx < k:
                out.append(item)
            else:
                penalized += 1
                adj = ScoredImage(
                    path=item.path,
                    score=item.score - soft_penalty * min(idx - k + 1, 3),
                    sharpness=item.sharpness,
                    exposure=item.exposure,
                    contrast=item.contrast,
                    saturation=item.saturation,
                    megapixels=item.megapixels,
                    noise=item.noise,
                    dof_separation=item.dof_separation,
                    clutter=item.clutter,
                    highlight_penalty=item.highlight_penalty,
                    oversat_penalty=item.oversat_penalty,
                    lighting_mood=item.lighting_mood,
                    phash=item.phash,
                )
                out.append(adj)
    out.sort(key=lambda x: x.score, reverse=True)
    return out, penalized


def pick_count(total: int, top_n: Optional[int], top_percent: Optional[float]) -> int:
    if total == 0:
        return 0
    if top_n is not None:
        return max(1, min(top_n, total))
    if top_percent is not None:
        return max(1, min(math.ceil(total * (top_percent / 100.0)), total))
    return max(1, math.ceil(total * 0.1))


def select_with_low_key_quota(candidates: list[ScoredImage], keep_count: int, low_key_quota_percent: float, low_key_exposure_max: float) -> list[ScoredImage]:
    if keep_count <= 0:
        return []
    if low_key_quota_percent <= 0:
        return candidates[:keep_count]

    quota = max(0, min(keep_count, int(round(keep_count * (low_key_quota_percent / 100.0)))))
    low_key = [c for c in candidates if c.exposure <= low_key_exposure_max]
    selected: list[ScoredImage] = low_key[:quota]
    selected_paths = {x.path for x in selected}
    for item in candidates:
        if len(selected) >= keep_count:
            break
        if item.path not in selected_paths:
            selected.append(item)
            selected_paths.add(item.path)
    return selected


def write_report(path: Path, scored: list[ScoredImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(
            "score\tsharpness\tdof_separation\tclutter\texposure\tlighting_mood\t"
            "contrast\tsaturation\thighlight_penalty\toversat_penalty\tnoise\t"
            "megapixels\tphash\tpath\n"
        )
        for item in scored:
            fh.write(
                f"{item.score:.6f}\t{item.sharpness:.6f}\t{item.dof_separation:.6f}\t"
                f"{item.clutter:.6f}\t{item.exposure:.6f}\t{item.lighting_mood:.6f}\t"
                f"{item.contrast:.6f}\t{item.saturation:.6f}\t{item.highlight_penalty:.6f}\t"
                f"{item.oversat_penalty:.6f}\t{item.noise:.6f}\t{item.megapixels:.3f}\t"
                f"{item.phash or ''}\t{item.path}\n"
            )


def write_selection_log(path: Path, scored: list[ScoredImage], selected: list[ScoredImage]) -> None:
    """Write per-file selection log into output folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_paths = {item.path for item in selected}
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(
            "rank\tselected\tscore\tsharpness\tdof_separation\tclutter\texposure\t"
            "lighting_mood\tcontrast\tsaturation\thighlight_penalty\toversat_penalty\t"
            "noise\tmegapixels\tpath\n"
        )
        for rank, item in enumerate(scored, start=1):
            is_selected = "1" if item.path in selected_paths else "0"
            fh.write(
                f"{rank}\t{is_selected}\t{item.score:.6f}\t{item.sharpness:.6f}\t"
                f"{item.dof_separation:.6f}\t{item.clutter:.6f}\t{item.exposure:.6f}\t"
                f"{item.lighting_mood:.6f}\t{item.contrast:.6f}\t{item.saturation:.6f}\t"
                f"{item.highlight_penalty:.6f}\t{item.oversat_penalty:.6f}\t"
                f"{item.noise:.6f}\t{item.megapixels:.3f}\t{item.path}\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select best shots by sharpness and image quality metrics."
    )
    parser.add_argument("--source", required=True, type=Path, help="Input photo folder")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--top-n", type=int, default=None, help="How many photos to keep")
    parser.add_argument("--top-percent", type=float, default=DEFAULT_TOP_PERCENT, help="How many percent of photos to keep (e.g. 15)")
    parser.add_argument("--copy-mode", choices=("flat", "preserve-tree"), default="flat", help="flat: copy all into one folder; preserve-tree: keep relative folders")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files without copying")
    parser.add_argument("--report", type=Path, default=None, help="Optional TSV report with per-file scores")
    parser.add_argument("--with-raw-pairs", action="store_true", default=True, help="When selected file is JPG/JPEG, also copy RAW with same basename")
    parser.add_argument("--no-with-raw-pairs", dest="with_raw_pairs", action="store_false", help="Disable copying RAW companions")
    parser.add_argument("--best-practice", action="store_true", default=DEFAULT_BEST_PRACTICE, help="Enable 3-step pipeline: technical floor + similar-shot dedup + final top")
    parser.add_argument("--no-best-practice", dest="best_practice", action="store_false", help="Disable best-practice pipeline")
    parser.add_argument("--min-sharpness", type=float, default=0.20, help="Technical floor for sharpness in best-practice mode (0..1)")
    parser.add_argument("--min-exposure", type=float, default=0.18, help="Technical floor for exposure in best-practice mode (0..1)")
    parser.add_argument("--min-contrast", type=float, default=0.12, help="Technical floor for contrast in best-practice mode (0..1)")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Lower means stricter similar-shot grouping in best-practice mode")
    parser.add_argument("--low-key-friendly", action="store_true", default=DEFAULT_LOW_KEY_FRIENDLY, help="Reduce penalty for intentionally dark (low-key) frames")
    parser.add_argument("--no-low-key-friendly", dest="low_key_friendly", action="store_false", help="Disable low-key-friendly behavior")
    parser.add_argument("--prefer-low-key", action="store_true", default=DEFAULT_PREFER_LOW_KEY, help="Add bonus for dark frames with good contrast")
    parser.add_argument("--no-prefer-low-key", dest="prefer_low_key", action="store_false", help="Disable low-key preference bonus")
    parser.add_argument("--creative-dark-rescue", action="store_true", default=DEFAULT_CREATIVE_DARK_RESCUE, help="Extra bonus for dark but structurally strong frames")
    parser.add_argument("--no-creative-dark-rescue", dest="creative_dark_rescue", action="store_false", help="Disable creative dark rescue bonus")
    parser.add_argument("--low-key-quota-percent", type=float, default=0.0, help="Reserve this percent of final picks for low-key frames")
    parser.add_argument("--low-key-exposure-max", type=float, default=0.58, help="Exposure threshold to classify low-key frames")
    parser.add_argument("--soft-dedup-penalty", type=float, default=0.04, help="Penalty per similar frame rank in best-practice mode")
    parser.add_argument("--cluster-keep-top-k", type=int, default=2, help="For large similar clusters keep top-K without penalty")
    parser.add_argument("--cluster-keep-top-k-min-size", type=int, default=5, help="Cluster size where top-K keeper rule applies")
    parser.add_argument("--workers", type=int, default=1, help="Process count for scoring (1 = no multiprocessing)")
    parser.add_argument("--weight-sharpness", type=float, default=None, help="Override sharpness score weight")
    parser.add_argument("--weight-exposure", type=float, default=None, help="Override exposure score weight")
    parser.add_argument("--weight-contrast", type=float, default=None, help="Override contrast score weight")
    parser.add_argument("--weight-saturation", type=float, default=None, help="Override saturation score weight")
    parser.add_argument("--weight-resolution", type=float, default=None, help="Override resolution score weight")
    parser.add_argument("--weight-noise-penalty", type=float, default=None, help="Override noise penalty weight")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs")
    parser.add_argument(
        "--split-by-subfolder",
        action="store_true",
        help="Process each immediate child folder of --source separately",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="{name}_best",
        help="Output folder template in --split-by-subfolder mode (supports {name})",
    )
    return parser


def run_selection(source: Path, output: Path, args: argparse.Namespace) -> int:
    weights = resolved_weights(args)

    images = list(iter_images(source, exclude_root=output))
    if not images:
        print(f"[{source.name}] No supported images found.")
        return 0

    scored, errors = score_all(
        images,
        weights,
        args.prefer_low_key,
        args.creative_dark_rescue,
        args.workers,
    )
    if not scored:
        print(f"[{source.name}] No images could be scored.")
        return 1

    scored.sort(key=lambda x: x.score, reverse=True)
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
            print("Best-practice floor removed everything; fallback to full scored set.")
            floored = scored
        candidates, soft_penalized = apply_soft_dedup(
            ranked=floored,
            similarity_threshold=args.similarity_threshold,
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
    print(f"[{source.name}] Selected: {len(selected)} files")
    selected_percent = (len(selected) / len(scored) * 100.0) if scored else 0.0
    print(f"[{source.name}] Selected percent: {selected_percent:.2f}%")
    if errors:
        print(f"[{source.name}] Scoring errors: {errors} (see warnings)")
    if args.dry_run:
        print(f"[{source.name}] Dry-run mode: no files copied")

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
        if args.dry_run:
            print(f"[{source.name}] Dry-run mode: report not written ({report_path})")
        else:
            write_report(report_path, scored)

    selection_log_path = output / "_selection_log.tsv"
    if args.dry_run:
        print(f"[{source.name}] Dry-run mode: selection log not written ({selection_log_path})")
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
        parser.error("Use only one of --top-n or --top-percent")
    if args.similarity_threshold <= 0:
        parser.error("--similarity-threshold must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.cluster_keep_top_k <= 0:
        parser.error("--cluster-keep-top-k must be > 0")
    if args.cluster_keep_top_k_min_size <= 0:
        parser.error("--cluster-keep-top-k-min-size must be > 0")
    if args.low_key_quota_percent < 0 or args.low_key_quota_percent > 100:
        parser.error("--low-key-quota-percent must be in [0, 100]")

    if not OPENCV_AVAILABLE:
        LOGGER.warning("OpenCV is unavailable; using fallback sharpness/noise metrics.")
    if args.best_practice and not IMAGEHASH_AVAILABLE:
        LOGGER.warning("imagehash is not installed; best-practice dedup will degrade to no pHash grouping.")

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
        # Skip generated folders from previous runs.
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
