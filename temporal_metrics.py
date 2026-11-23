import argparse
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import disk


def read_mask(path):
    with Image.open(path) as image:
        return np.array(image)


def iou(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} / {gt.shape}")
    union = (pred | gt).sum()
    return (pred & gt).sum() / union if union else 1.0


def _seg2bmap(seg):
    seg = np.asarray(seg, dtype=bool)
    east = np.zeros_like(seg)
    south = np.zeros_like(seg)
    southeast = np.zeros_like(seg)
    east[:, :-1] = seg[:, 1:]
    south[:-1, :] = seg[1:, :]
    southeast[:-1, :-1] = seg[1:, 1:]

    boundary = (seg ^ east) | (seg ^ south) | (seg ^ southeast)
    boundary[-1, :] = seg[-1, :] ^ east[-1, :]
    boundary[:, -1] = seg[:, -1] ^ south[:, -1]
    boundary[-1, -1] = False
    return boundary


@lru_cache(maxsize=32)
def _boundary_disk(shape):
    radius = int(np.ceil(0.008 * np.linalg.norm(shape)))
    return disk(radius)


def boundary_f(pred, gt):
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} / {gt.shape}")
    if pred.ndim != 2 or pred.size == 0:
        raise ValueError("Boundary F-score requires non-empty 2D masks")
    pred_boundary = _seg2bmap(pred)
    gt_boundary = _seg2bmap(gt)
    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    kernel = _boundary_disk(pred.shape)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel)
    gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel)
    precision = (pred_boundary * gt_dilated).sum() / float(n_pred)
    recall = (gt_boundary * pred_dilated).sum() / float(n_gt)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def jf(pred, gt):
    return (iou(pred, gt) + boundary_f(pred, gt)) / 2


def lat_cov(scores):
    valid = [(b, a) for b, a in scores if b > 0.3]  # valid frames
    if not valid:
        return np.nan, np.nan  # Exclude video-object pairs with no valid frames at all.
    collapsed = [i for i, (b, a) in enumerate(valid, 1) if a <= b * 0.5] # collapsed frames
    lat = collapsed[0] / len(valid) if collapsed else 1.0 # Latency
    cov = len(collapsed) / len(valid) # Coverage
    return lat, cov


def evaluate_video(task):
    video, args = task
    gt_dir = args.gt / video
    frames = sorted(gt_dir.glob('*.png'), key=lambda p: int(p.stem))
    objects = [int(i) for i in np.unique(read_mask(frames[0])) if i not in (0, 255)]
    if not objects:
        raise ValueError(f"No objects in the first frame: {video}")
    if not args.include_ends:
        frames = frames[1:-1]

    scores = {obj: [] for obj in objects}
    scores20 = {obj: [] for obj in objects}
    for frame in frames:
        gt = read_mask(frame)
        clean = read_mask(args.benign / video / frame.name)
        attack = read_mask(args.attack / video / frame.name)
        # attack20 requires a separate mid-video attack starting at frame 20.
        delayed = None
        if args.attack20 is not None and int(frame.stem) >= 20:
            delayed = read_mask(args.attack20 / video / frame.name)
        for obj in objects:
            b = jf(clean == obj, gt == obj)
            scores[obj].append((b, jf(attack == obj, gt == obj)))
            if delayed is not None:
                scores20[obj].append((b, jf(delayed == obj, gt == obj)))
            # Each item is (benign J&F, attacked J&F).

    results = []
    for obj in objects:
        lat, cov = lat_cov(scores[obj])
        if np.isnan(lat):
            continue
        values = [lat, cov]
        if args.attack20 is not None:
            values.append(lat_cov(scores20[obj])[1])
        results.append((video, obj, values))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True, help='GT masks')
    parser.add_argument('--benign', type=Path, required=True, help='clean prediction')
    parser.add_argument('--attack', type=Path, required=True, help='attacked prediction')
    parser.add_argument('--attack20', type=Path, help='attacked prediction starting at frame 20')
    parser.add_argument('-n', '--num-processes', type=int, default=16)
    parser.add_argument('--include-ends', action='store_true')
    args = parser.parse_args()

    videos = sorted(p.name for p in args.gt.iterdir() if p.is_dir())
    with Pool(args.num_processes) as pool:
        results = pool.map(evaluate_video, [(video, args) for video in videos])

    names = ['Lat', 'Cov'] + (['Cov20'] if args.attack20 is not None else [])
    print('Frame score: J&F = (IoU + boundary F) / 2')
    print('Video / Object / ' + ' / '.join(names) + ' (%)')
    all_values = []
    for rows in results:
        for video, obj, values in rows:
            print(video, obj, ' / '.join(f'{v * 100:.1f}' for v in values))
            all_values.append(values)
    means = []
    for i in range(len(names)):
        values = [row[i] for row in all_values if not np.isnan(row[i])]
        means.append(np.mean(values) * 100 if values else np.nan)
    print(f'Evaluated pairs: {len(all_values)}')
    print('Global: ' + ' / '.join(f'{name}={v:.1f}' for name, v in zip(names, means)))


if __name__ == '__main__':
    main()
