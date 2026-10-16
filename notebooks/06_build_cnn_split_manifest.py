"""Build a deterministic, duplicate-group-aware CNN split manifest.

This script does not delete or modify source images. Exact duplicates and
perceptual-hash candidates (Hamming distance <= 4) are kept in the same split.
Run from the repository root:
    python notebooks/06_build_cnn_split_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "fire_dataset"
DEFAULT_OUTPUT = ROOT / "notebooks" / "plots" / "cnn_split_manifest_rebuilt.csv"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff"}
PHASH_DISTANCE = 4


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def image_phash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def read_records(dataset_dir: Path) -> list[dict]:
    records = []
    for label in ("fire", "no_fire"):
        folder = dataset_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Required class folder is missing: {folder}")
        for path in sorted(p for p in folder.rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSIONS):
            try:
                raw = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
            except Exception:
                image = None
            if image is None:
                raise ValueError(f"Unreadable image; repair or remove it before splitting: {path}")
            records.append({
                "relative_path": path.relative_to(ROOT).as_posix(),
                "label": label,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "phash": image_phash(image),
            })
    if not records:
        raise ValueError(f"No supported images found under {dataset_dir}")
    return records


def assign_duplicate_groups(records: list[dict], max_distance: int = PHASH_DISTANCE) -> np.ndarray:
    """Union exact duplicates and near-duplicate candidates into connected groups."""
    n = len(records)
    uf = UnionFind(n)

    # Exact byte-identical files are always grouped.
    by_sha: dict[str, int] = {}
    for i, row in enumerate(records):
        prior = by_sha.setdefault(row["sha256"], i)
        uf.union(prior, i)

    hashes = np.asarray([row["phash"] for row in records], dtype=np.uint64)
    popcount = np.asarray([int(v).bit_count() for v in range(256)], dtype=np.uint8)
    block_size = 128
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        xor_values = np.bitwise_xor(hashes[start:end, None], hashes[None, :])
        distances = popcount[
            xor_values.view(np.uint8).reshape(end - start, n, 8)
        ].sum(axis=2)
        rows, cols = np.where(
            (distances <= max_distance)
            & (np.arange(start, end)[:, None] < np.arange(n)[None, :])
        )
        for local_i, j in zip(rows.tolist(), cols.tolist()):
            uf.union(start + local_i, int(j))

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(uf.find(i), []).append(i)

    # A duplicate candidate group spanning both labels indicates possible label
    # corruption. Stop instead of silently accepting conflicting labels.
    conflicts = [
        [records[i]["relative_path"] for i in members]
        for members in components.values()
        if len({records[i]["label"] for i in members}) > 1
    ]
    if conflicts:
        preview = conflicts[:3]
        raise ValueError(
            "Duplicate-candidate groups span fire/no_fire labels. Review these "
            f"groups before training; examples: {json.dumps(preview)}"
        )

    ordered_components = sorted(
        components.values(),
        key=lambda members: min(records[i]["relative_path"] for i in members),
    )
    group_ids = np.empty(n, dtype=object)
    for group_number, members in enumerate(ordered_components, start=1):
        group_id = f"G{group_number:05d}"
        for i in members:
            group_ids[i] = group_id
    return group_ids


def make_manifest(records: list[dict], group_ids: np.ndarray, seed: int = 42) -> pd.DataFrame:
    labels = np.asarray([0 if r["label"] == "fire" else 1 for r in records])
    indexes = np.arange(len(records))

    # Approximate 70/15/15 split while ensuring a connected duplicate group
    # can never cross train, validation, and test.
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(indexes, labels, group_ids))

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(train_val_idx, labels[train_val_idx], group_ids[train_val_idx])
    )
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    split_by_index = {}
    for split, selected in (
        ("train", train_idx), ("validation", val_idx), ("test", test_idx)
    ):
        for i in selected:
            split_by_index[int(i)] = split

    group_sizes = pd.Series(group_ids).value_counts().to_dict()
    rows = []
    for i, record in enumerate(records):
        rows.append({
            "relative_path": record["relative_path"],
            "label": record["label"],
            "group_id": group_ids[i],
            "split": split_by_index[i],
            "group_size": int(group_sizes[group_ids[i]]),
            "candidate_group": bool(group_sizes[group_ids[i]] > 1),
        })
    manifest = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)

    group_split_counts = manifest.groupby("group_id")["split"].nunique()
    if (group_split_counts > 1).any():
        raise AssertionError("Internal error: duplicate group crosses split boundaries.")
    if manifest["relative_path"].duplicated().any():
        raise AssertionError("Internal error: duplicate manifest paths.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=PHASH_DISTANCE)
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 16:
        parser.error("--phash-distance must be between 0 and 16")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {args.output}. "
            "Choose another --output path or move the existing file intentionally."
        )

    print(f"Scanning images under {args.dataset}...")
    records = read_records(args.dataset)
    print(f"Images scanned: {len(records)}")
    print(f"Grouping exact and perceptual duplicates (pHash distance <= {args.phash_distance})...")
    group_ids = assign_duplicate_groups(records, args.phash_distance)
    manifest = make_manifest(records, group_ids, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print("\nCounts by split and class:")
    print(pd.crosstab(manifest["split"], manifest["label"]))
    print(f"Unique duplicate groups: {manifest['group_id'].nunique()}")
    print(f"Images in candidate groups: {int(manifest['candidate_group'].sum())}")
    print(f"Manifest written: {args.output}")
    print("Source images were not deleted or modified.")


if __name__ == "__main__":
    main()
