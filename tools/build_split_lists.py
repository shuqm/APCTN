#!/usr/bin/env python3
"""Build APCTN split-list files from CWT image folders.

Expected image layout:
    data/<dataset>/<domain>/images/<class_name>/*.png

The generated split files contain lines in the form:
    <domain>/images/<class_name>/<image_file> <class_id>
"""

import argparse
import random
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect(domain_dir):
    images_dir = domain_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError("Missing images directory: {}".format(images_dir))
    classes = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    records = []
    for class_id, class_dir in enumerate(classes):
        for image in sorted(class_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                rel = image.relative_to(domain_dir.parent).as_posix()
                records.append((rel, class_id))
    if not records:
        raise RuntimeError("No images found under {}".format(images_dir))
    return records, classes


def balanced_fewshot(records, total, seed):
    rng = random.Random(seed)
    by_class = {}
    for rec in records:
        by_class.setdefault(rec[1], []).append(rec)
    classes = sorted(by_class)
    for c in classes:
        rng.shuffle(by_class[c])

    selected = []
    cursor = {c: 0 for c in classes}
    while len(selected) < total:
        progress = False
        for c in classes:
            if len(selected) >= total:
                break
            if cursor[c] < len(by_class[c]):
                selected.append(by_class[c][cursor[c]])
                cursor[c] += 1
                progress = True
        if not progress:
            raise ValueError("Requested {} labeled samples but only {} images exist".format(total, len(records)))
    return selected


def write_list(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rel, label in records:
            f.write("{} {}\n".format(rel, label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["case1_shaft", "case2_gear"])
    parser.add_argument("--domain", required=True, help="e.g. 200rpm, 250rpm, 400rpm, 1000rpm")
    parser.add_argument("--fewshot", required=True, type=int, help="total labeled source samples")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split-root", default="./data/splits")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    domain_dir = Path(args.data_root) / args.dataset / args.domain
    records, classes = collect(domain_dir)
    labeled = balanced_fewshot(records, args.fewshot, args.seed)
    labeled_set = set(labeled)
    unlabeled = [r for r in records if r not in labeled_set]

    out = Path(args.split_root) / args.dataset
    write_list(out / (args.domain + ".txt"), records)
    write_list(out / (args.domain + "_labeled_{}.txt".format(args.fewshot)), labeled)
    write_list(out / (args.domain + "_unlabeled_{}.txt".format(args.fewshot)), unlabeled)

    print("Dataset:", args.dataset)
    print("Domain:", args.domain)
    print("Classes:", [p.name for p in classes])
    print("Total images:", len(records))
    print("Labeled images:", len(labeled))
    print("Unlabeled images:", len(unlabeled))
    print("Split files written to:", out)


if __name__ == "__main__":
    main()
