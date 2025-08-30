"""
Build a tiny 'mini_ucf' dataset from a partially-downloaded UCF101 folder.

Handles the annoying case where train/val/test folders only include a few
classes, but the CSVs list ALL classes. We filter rows to:
  - labels that actually exist under split folders, and
  - files that actually exist on disk.

Outputs:
data/mini_ucf/
  train/<Class>/*.avi
  val/<Class>/*.avi
  [optional] test/<Class>/*.avi

Defaults: 3 classes, 8 train vids/class, 4 val vids/class.
"""

import argparse, csv, os, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple

RNG = random.Random(42)


def read_csv_table(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def scan_available_classes(split_root: Path) -> List[str]:
    if not split_root.exists():
        return []
    return sorted([d.name for d in split_root.iterdir() if d.is_dir()])


def resolve_clip_path(rel_or_abs: str, ds_root: Path) -> Path:
    """
    CSV clip_path may look like '/train/Swing/v_...avi' or 'train/Swing/v_...avi'.
    Resolve it robustly under ds_root.
    """
    rel = rel_or_abs.lstrip("/\\")
    return (ds_root / rel).resolve()


def build_index(
    rows: List[Dict],
    ds_root: Path,
    split: str,
    allowed_classes: List[str],
    allowed_exts: Tuple[str, ...],
):
    """Return dict[label] -> list[Path] of existing files for this split only."""
    by_label: Dict[str, List[Path]] = {}
    missing = 0
    skipped_label = 0

    split_prefix = f"{split}/"
    for r in rows:
        label = r["label"]
        # only keep rows whose label exists in the current split folder set
        if label not in allowed_classes:
            skipped_label += 1
            continue

        clip_path = r.get("clip_path") or ""
        # some CSVs don’t include the split in the path; ensure it does
        if not clip_path.lstrip("/\\").startswith(split_prefix):
            # try to fix: if it starts with another split, skip
            parts = clip_path.lstrip("/\\").split("/", 1)
            if parts and parts[0] in {"train", "val", "test"} and parts[0] != split:
                continue
            # otherwise, prepend the split and label
            clip_path = f"{split}/{label}/{r['clip_name']}.avi"

        p = resolve_clip_path(clip_path, ds_root)
        if not p.exists() or p.suffix.lower() not in allowed_exts:
            missing += 1
            continue

        by_label.setdefault(label, []).append(p)

    return by_label, missing, skipped_label


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            return
        os.symlink(src, dst)
        return
    except Exception:
        pass
    try:
        if dst.exists():
            return
        os.link(src, dst)
        return
    except Exception:
        pass
    if not dst.exists():
        shutil.copy2(src, dst)


def sample(items: List[Path], k: int) -> List[Path]:
    if k <= 0 or k >= len(items):
        return list(items)
    return RNG.sample(items, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="Path to UCF101 root with partial train/val[/test] and CSVs",
    )
    ap.add_argument("--dst", default="data/mini_ucf", help="Output folder")
    ap.add_argument(
        "--classes",
        default="",
        help="Comma-separated class names to keep; empty means auto-pick from available",
    )
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--train-per-class", type=int, default=8)
    ap.add_argument("--val-per-class", type=int, default=4)
    ap.add_argument("--test-per-class", type=int, default=4)
    ap.add_argument(
        "--include-test",
        action="store_true",
        help="Also build test split if test/ exists",
    )
    ap.add_argument(
        "--ext-keep", default=".avi,.mp4,.mov,.mkv", help="Allowed video extensions"
    )
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    # Roots and CSVs
    train_root = src / "train"
    val_root = src / "val"
    test_root = src / "test"
    train_csv = src / "train.csv"
    val_csv = src / "val.csv"
    test_csv = src / "test.csv"

    # sanity
    if (
        not train_root.exists()
        or not train_csv.exists()
        or not val_root.exists()
        or not val_csv.exists()
    ):
        raise SystemExit(
            "Expected train/ val/ folders and train.csv/val.csv under --src. Your path looks wrong."
        )

    # available classes on disk per split
    avail_train = set(scan_available_classes(train_root))
    avail_val = set(scan_available_classes(val_root))
    avail_test = set(scan_available_classes(test_root)) if test_root.exists() else set()

    # decide final class list
    if args.classes.strip():
        keep = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        common = sorted(avail_train & avail_val)  # ensure present in both splits
        if not common:
            raise SystemExit(
                "No overlapping classes between train/ and val/. Download at least one class in both."
            )
        keep = RNG.sample(common, k=min(args.num_classes, len(common)))

    print("Classes chosen:", keep)

    # load CSVs
    train_rows = read_csv_table(train_csv)
    val_rows = read_csv_table(val_csv)
    test_rows = (
        read_csv_table(test_csv)
        if args.include_test and test_root.exists() and test_csv.exists()
        else []
    )

    allowed_exts = tuple(
        x.strip().lower() for x in args.ext_keep.split(",") if x.strip()
    )

    # build indices filtered by existing classes and files
    train_idx, miss_tr, skiplab_tr = build_index(
        train_rows, src, "train", keep, allowed_exts
    )
    val_idx, miss_va, skiplab_va = build_index(val_rows, src, "val", keep, allowed_exts)
    if test_rows:
        # only keep classes present on disk in test split too
        keep_test = [c for c in keep if c in avail_test]
        test_idx, miss_te, skiplab_te = build_index(
            test_rows, src, "test", keep_test, allowed_exts
        )
    else:
        test_idx, miss_te, skiplab_te = {}, 0, 0

    # report filtering
    print(f"[train] kept labels: {sorted(train_idx.keys())}")
    print(
        f"[train] skipped rows wrong/missing label: {skiplab_tr}, missing files/ext: {miss_tr}"
    )
    print(f"[val]   kept labels: {sorted(val_idx.keys())}")
    print(
        f"[val]   skipped rows wrong/missing label: {skiplab_va}, missing files/ext: {miss_va}"
    )
    if test_rows:
        print(f"[test]  kept labels: {sorted(test_idx.keys())}")
        print(
            f"[test]  skipped rows wrong/missing label: {skiplab_te}, missing files/ext: {miss_te}"
        )

    # create output and populate
    for phase, index, k in [
        ("train", train_idx, args.train_per_class),
        ("val", val_idx, args.val_per_class),
    ] + ([("test", test_idx, args.test_per_class)] if test_rows else []):
        for cls in keep if phase != "test" else sorted(test_idx.keys()):
            files = index.get(cls, [])
            if not files:
                print(f"[warn] No files for class {cls} in {phase}, skipping.")
                continue
            out_dir = dst / phase / cls
            ensure_dir(out_dir)
            for i, src_file in enumerate(sample(files, k), 1):
                dst_name = f"{cls.lower()}_{i:04d}{src_file.suffix.lower()}"
                link_or_copy(src_file, out_dir / dst_name)

    # summary
    print("\nMini dataset created at:", dst)
    for phase in ["train", "val"] + (["test"] if test_rows else []):
        total = 0
        for cls in sorted((dst / phase).iterdir()):
            if not cls.is_dir():
                continue
            count = len(list(cls.glob("*")))
            total += count
            print(f"{phase:<5} {cls.name:<20} {count:4d}")
        print(f"{phase:<5} total: {total}")
    print("\nPoint training.config.yaml data.video.root to:", dst)


if __name__ == "__main__":
    main()
