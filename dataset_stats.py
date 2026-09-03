"""
dataset_stats.py — 합성 데이터셋 통계 보고 (작업 5)

data/det/synth/ 를 읽어 아래를 산출한다:
  · 총 이미지 수 / 총 라벨 수 / 이미지당 평균
  · 라벨 박스 크기 분포 (사분위수)
  · 자세별 분포 (서있음 / 쓰러짐 / 부분가림 / 보정표본)
  · 이론 픽셀값과의 정합성 (GSD 기반)
결과를 dataset_stats.json 으로 저장하고 콘솔에 표로 출력.

자세 판별: 인스턴스 인덱스 순서가 place_persons.py 의 배치 순서와 같다는 규약을 이용
  (서있음 N_STANDING → 가림 N_OCCLUDED → 쓰러짐 → 보정표본 CALIB)

실행: python dataset_stats.py [--root dataset_synth]
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

PERSON_LEN_M = 1.7      # 쓰러진 사람 장축
PERSON_W_M = 0.5

# place_persons.py 기본값과 일치해야 함
N_STANDING, N_OCCLUDED, N_LYING, N_CALIB = 4, 3, 9, 2


def pose_labels(n_instances):
    seq = (["standing"] * N_STANDING
           + ["lying_occluded"] * N_OCCLUDED
           + ["lying"] * (N_LYING - N_OCCLUDED)
           + ["calib"] * N_CALIB)
    if len(seq) < n_instances:
        seq += ["unknown"] * (n_instances - len(seq))
    return seq[:n_instances]


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(BASE_DIR / "data" / "det" / "synth"))
    args = ap.parse_args()
    root = Path(args.root)

    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    cmap = json.loads((BASE_DIR / "configs" / "seg_color_map.json").read_text(encoding="utf-8"))
    by_idx = sorted(cmap["person"], key=lambda e: e["instance_index"])
    poses = pose_labels(len(by_idx))
    pose_of = {e["name"]: poses[i] for i, e in enumerate(by_idx)}
    rgb2name = {tuple(e["rgb"]): e["name"] for e in cmap["person"]}

    # ── 라벨 파일 통계 ──
    label_files = sorted((root / "yolo" / "labels").rglob("*.txt"))
    img_w, img_h = info["image_size"]
    n_labels = 0
    widths, heights, longs = [], [], []
    per_image = []
    empty = 0
    for lf in label_files:
        lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
        per_image.append(len(lines))
        if not lines:
            empty += 1
        for l in lines:
            _, cx, cy, w, h = l.split()
            wpx, hpx = float(w) * img_w, float(h) * img_h
            widths.append(wpx); heights.append(hpx); longs.append(max(wpx, hpx))
            n_labels += 1

    # ── 자세별 분포 (seg 원본에서 개체 단위 집계) ──
    pose_counts = Counter()
    pose_long = {}
    for seg_p in sorted((root / "raw").glob("seg_*.png")):
        idx = seg_p.stem.split("_")[1]
        meta = json.loads((root / "raw" / f"meta_{idx}.json").read_text(encoding="utf-8"))
        nadir = meta["cam_pitch_deg"] == -90.0
        seg = imread_u(seg_p)
        for color, name in rgb2name.items():
            c = np.array(color, dtype=np.uint8)
            mask = cv2.inRange(seg, c, c)
            if int(mask.sum() / 255) < 100:
                continue
            ys, xs = np.where(mask > 0)
            L = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            p = pose_of[name]
            pose_counts[p] += 1
            if nadir:
                pose_long.setdefault(p, []).append((meta["altitude_m"], L))

    q = lambda a: np.percentile(a, [0, 25, 50, 75, 100]).round(1).tolist()
    stats = {
        "images": len(label_files),
        "labels_total": n_labels,
        "labels_per_image_mean": round(n_labels / max(len(label_files), 1), 2),
        "empty_images": empty,
        "box_long_side_px_quartiles": q(longs) if longs else [],
        "box_width_px_quartiles": q(widths) if widths else [],
        "box_height_px_quartiles": q(heights) if heights else [],
        "instance_counts_by_pose": dict(pose_counts),
        "collection": {k: info[k] for k in
                       ("altitudes_m", "camera_pitches_deg", "fov_degrees", "image_size")},
    }

    # ── 이론 정합성 (수직 하방 프레임만) ──
    consistency = {}
    for p, arr in pose_long.items():
        rows = []
        for alt, L in arr:
            gsd = (2 * alt * math.tan(math.radians(info["fov_degrees"] / 2))) / img_w * 100
            rows.append(L / (PERSON_LEN_M * 100 / gsd))
        consistency[p] = {"samples": len(rows), "ratio_median": round(float(np.median(rows)), 3)}
    stats["theory_ratio_nadir"] = consistency

    (root / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 출력 ──
    print("=== 데이터셋 통계 ===")
    print(f"이미지 {stats['images']}장 | 라벨 {stats['labels_total']}개 | "
          f"이미지당 평균 {stats['labels_per_image_mean']} | 라벨없음 {empty}장")
    c = stats["collection"]
    print(f"수집 조건: 고도 {c['altitudes_m']}m, 카메라각 {c['camera_pitches_deg']}°, "
          f"FOV {c['fov_degrees']}°, {c['image_size'][0]}x{c['image_size'][1]}")
    print()
    print("박스 크기 분포 (px)   최소   Q1   중앙   Q3   최대")
    for nm, key in (("긴 변", "box_long_side_px_quartiles"),
                    ("폭", "box_width_px_quartiles"),
                    ("높이", "box_height_px_quartiles")):
        v = stats[key]
        print(f"  {nm:6} {v[0]:>7.1f} {v[1]:>6.1f} {v[2]:>6.1f} {v[3]:>6.1f} {v[4]:>7.1f}")
    print()
    print("자세별 개체 등장 수")
    for p, n in sorted(pose_counts.items(), key=lambda x: -x[1]):
        print(f"  {p:16} {n:>6}")
    print()
    print("이론 정합성 (수직하방, 긴변/이론 1.7m)")
    for p, d in consistency.items():
        print(f"  {p:16} 표본 {d['samples']:>4}  비율 {d['ratio_median']:.2f}")
    print(f"\n저장: {root / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
