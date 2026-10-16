from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "06_build_cnn_split_manifest.py"
SPEC = importlib.util.spec_from_file_location("cnn_split_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_exact_and_near_duplicates_are_grouped():
    records = [
        {"relative_path": "a.jpg", "label": "fire", "sha256": "same", "phash": 0},
        {"relative_path": "b.jpg", "label": "fire", "sha256": "same", "phash": 0},
        {"relative_path": "c.jpg", "label": "fire", "sha256": "third", "phash": 3},
        {"relative_path": "d.jpg", "label": "fire", "sha256": "fourth", "phash": (1 << 64) - 1},
    ]
    groups = builder.assign_duplicate_groups(records, max_distance=4)
    assert groups[0] == groups[1]
    assert groups[0] == groups[2]
    assert groups[3] != groups[0]


def test_conflicting_labels_in_duplicate_group_are_rejected():
    records = [
        {"relative_path": "fire/a.jpg", "label": "fire", "sha256": "same", "phash": 0},
        {"relative_path": "no_fire/a.jpg", "label": "no_fire", "sha256": "same", "phash": 0},
    ]
    with pytest.raises(ValueError, match="span fire/no_fire"):
        builder.assign_duplicate_groups(records, max_distance=4)


def test_manifest_keeps_groups_inside_one_split():
    records = []
    group_ids = []
    for label in ("fire", "no_fire"):
        for i in range(70):
            # Make several two-image groups and leave the remainder as singletons.
            group_id = f"{label}-pair-{i // 2}" if i < 28 else f"{label}-single-{i}"
            records.append({
                "relative_path": f"{label}/{i:03d}.jpg",
                "label": label,
            })
            group_ids.append(group_id)
    manifest = builder.make_manifest(records, np.asarray(group_ids, dtype=object), seed=42)
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert not (manifest.groupby("group_id")["split"].nunique() > 1).any()
    assert not manifest["relative_path"].duplicated().any()
    assert set(manifest["label"]) == {"fire", "no_fire"}
