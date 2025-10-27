#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 mmseg 数据集的图像/标注匹配问题：
- 同名前缀匹配（img 与 ann）
- 缺失样本（孤儿图像/孤儿标注）
- 图像与掩膜尺寸不一致
- 掩膜通道数检查（应为单通道）
- 掩膜像素取值检查（是否含 ignore、是否超出类别数）
用法示例：
    python check_dataset_matching.py \
        --root datasets/tongue_seg_v0 \
        --splits train val \
        --img-suffixes .jpg .jpeg .png \
        --mask-suffixes .png .bmp \
        --num-classes 2 \
        --ignore-index 255 \
        --report report.csv
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

from PIL import Image
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description='Check mmseg dataset matching')
    p.add_argument('--root', required=True, type=str,
                   help='dataset root, e.g. datasets/tongue_seg_v0')
    p.add_argument('--img-dirname', default='img_dir', type=str)
    p.add_argument('--ann-dirname', default='ann_dir', type=str)
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                   help='subfolders under img_dir/ and ann_dir/')
    p.add_argument('--img-suffixes', nargs='+', default=['.jpg', '.jpeg', '.png'],
                   help='allowed image suffixes (case-insensitive)')
    p.add_argument('--mask-suffixes', nargs='+', default=['.png', '.bmp', '.tif', '.tiff'],
                   help='allowed mask suffixes (case-insensitive)')
    p.add_argument('--recursive', action='store_true', help='scan recursively under split folders')
    p.add_argument('--num-classes', type=int, default=None,
                   help='expected classes (including background). If set, check mask values in [0, num_classes-1] plus ignore-index.')
    p.add_argument('--ignore-index', type=int, default=255)
    p.add_argument('--report', type=str, default=None, help='optional CSV path to dump detailed rows')
    return p.parse_args()

def list_files(folder: Path, suffixes: Set[str], recursive: bool) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    else:
        files = [p for p in folder.glob('*') if p.is_file() and p.suffix.lower() in suffixes]
    return files

def build_stem_map(files: List[Path]) -> Dict[str, List[Path]]:
    stems: Dict[str, List[Path]] = {}
    for p in files:
        stems.setdefault(p.stem, []).append(p)
    return stems

def image_size(p: Path) -> Tuple[int, int, int]:
    # returns (H, W, C)
    with Image.open(p) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
    else:
        raise ValueError(f'Unsupported image ndim={arr.ndim} for {p}')
    return h, w, c

def mask_values(p: Path) -> Tuple[bool, int, int, np.ndarray]:
    """Return (is_single_channel, min, max, unique_values_clip)
    unique_values_clip: up to first 32 unique values (sorted)
    """
    with Image.open(p) as im:
        arr = np.array(im)
    if arr.ndim == 2:
        is_single = True
        vals = arr
    elif arr.ndim == 3:
        is_single = False
        # if RGB mask, collapse to one channel to inspect distribution anyway
        if arr.shape[2] >= 1:
            vals = arr[:, :, 0]
        else:
            vals = arr.reshape(arr.shape[0], arr.shape[1])
    else:
        is_single = False
        vals = arr
    vmin = int(vals.min())
    vmax = int(vals.max())
    uniq = np.unique(vals)
    if uniq.size > 32:
        uniq_clip = uniq[:32]
    else:
        uniq_clip = uniq
    return is_single, vmin, vmax, uniq_clip

def check_split(root: Path, img_dirname: str, ann_dirname: str, split: str,
                img_suffixes: Set[str], mask_suffixes: Set[str], recursive: bool,
                num_classes: int, ignore_index: int):
    img_dir = root / img_dirname / split
    ann_dir = root / ann_dirname / split
    imgs = list_files(img_dir, img_suffixes, recursive)
    masks = list_files(ann_dir, mask_suffixes, recursive)

    img_stems = build_stem_map(imgs)
    mask_stems = build_stem_map(masks)

    img_only = sorted(set(img_stems.keys()) - set(mask_stems.keys()))
    mask_only = sorted(set(mask_stems.keys()) - set(img_stems.keys()))
    both = sorted(set(img_stems.keys()) & set(mask_stems.keys()))

    dup_img_stems = [k for k, v in img_stems.items() if len(v) > 1]
    dup_mask_stems = [k for k, v in mask_stems.items() if len(v) > 1]

    problems = []
    # basic summary
    summary = {
        'split': split,
        'images_found': len(imgs),
        'masks_found': len(masks),
        'pairs': len(both),
        'img_only': len(img_only),
        'mask_only': len(mask_only),
        'dup_img_stems': len(dup_img_stems),
        'dup_mask_stems': len(dup_mask_stems),
    }

    # detail checks for matched pairs
    for stem in both:
        # choose the first when duplicates exist (but record as problem)
        img_p = sorted(img_stems[stem])[0]
        mask_p = sorted(mask_stems[stem])[0]
        try:
            ih, iw, ic = image_size(img_p)
        except Exception as e:
            problems.append((split, stem, 'bad_image', str(img_p), str(e)))
            continue
        try:
            mh, mw, mc = image_size(mask_p)
        except Exception as e:
            problems.append((split, stem, 'bad_mask', str(mask_p), str(e)))
            continue

        if (ih, iw) != (mh, mw):
            problems.append((split, stem, 'size_mismatch',
                             f'{img_p} ({ih}x{iw}x{ic})',
                             f'{mask_p} ({mh}x{mw}x{mc})'))

        # mask channel/value sanity
        is_single, vmin, vmax, uniq = mask_values(mask_p)
        if not is_single:
            problems.append((split, stem, 'mask_not_single_channel', str(mask_p),
                             f'shape={mh}x{mw}x{mc}'))
        # value range checks
        if num_classes is not None:
            # allowed values: [0, num_classes-1] U {ignore}
            allowed = set(range(0, num_classes))
            allowed.add(ignore_index)
            # quick check via min/max; then sample uniques (capped)
            if not (vmin in allowed and vmax in allowed):
                # more strict: if any uniq outside allowed
                outside = [int(x) for x in uniq if int(x) not in allowed]
                if outside:
                    problems.append((split, stem, 'mask_value_out_of_range',
                                     str(mask_p),
                                     f'min={vmin}, max={vmax}, out={outside[:16]}...'))
        else:
            # 如果未指定类别数，至少给个提醒
            if vmax > 255 or vmin < 0:
                problems.append((split, stem, 'mask_value_suspicious',
                                 str(mask_p), f'min={vmin}, max={vmax}'))

    # record orphans & duplicates
    for stem in img_only:
        problems.append((split, stem, 'mask_missing', str(sorted(img_stems[stem])[0]), ''))
    for stem in mask_only:
        problems.append((split, stem, 'image_missing', str(sorted(mask_stems[stem])[0]), ''))
    for stem in dup_img_stems:
        problems.append((split, stem, 'duplicate_images', ', '.join(map(str, sorted(img_stems[stem]))), ''))
    for stem in dup_mask_stems:
        problems.append((split, stem, 'duplicate_masks', ', '.join(map(str, sorted(mask_stems[stem]))), ''))

    return summary, problems

def main():
    args = parse_args()
    root = Path(args.root)
    img_suffixes = set([s.lower() for s in args.img_suffixes])
    mask_suffixes = set([s.lower() for s in args.mask_suffixes])

    all_summary = []
    all_problems = []

    print('=== Dataset root:', root)
    for split in args.splits:
        summary, problems = check_split(
            root, args.img_dirname, args.ann_dirname, split,
            img_suffixes, mask_suffixes, args.recursive,
            args.num_classes if args.num_classes is not None else None,
            args.ignore_index
        )
        all_summary.append(summary)
        all_problems.extend(problems)

        print(f'\n[{split}] images={summary["images_found"]}, masks={summary["masks_found"]}, '
              f'pairs={summary["pairs"]}')
        print(f'       img_only={summary["img_only"]}, mask_only={summary["mask_only"]}, '
              f'dup_img={summary["dup_img_stems"]}, dup_mask={summary["dup_mask_stems"]}')
        if problems:
            # 打印前 10 条问题
            print('       sample problems:')
            for row in problems[:10]:
                print('         -', row)

    if args.report:
        rpt = Path(args.report)
        rpt.parent.mkdir(parents=True, exist_ok=True)
        with rpt.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['split', 'stem', 'issue', 'path_a', 'path_b_or_detail'])
            for row in all_problems:
                w.writerow(row)
        print(f'\nDetailed report saved to: {rpt.resolve()}')

    # 退出码：有问题则返回非零
    if any(all_problems):
        print('\nSummary: FOUND issues. Please check the report above.')
        exit(1)
    else:
        print('\nSummary: No issues found.')
        exit(0)

if __name__ == '__main__':
    main()
