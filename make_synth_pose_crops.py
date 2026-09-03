"""
make_synth_pose_crops.py — 합성 데이터에서 자세별 크롭 생성

왜 합성인가
  실사(NOMAD)로 자세 분류를 두 번 시도해 두 번 실패했다.
    · 탐지기를 2클래스로  → 쓰러짐 재현율 0.153
    · 별도 분류기         → 0.297 (2단계 결합 0.253)
  실사의 한계로 보이는 것들:
    · 쓰러진 표본이 검증 209개뿐
    · `Hiding (Laying)` 은 수풀에 가려 몸이 절반만 보인다
    · 활동 라벨이 프레임 구간 단위라 경계가 흐리다

  합성은 이 셋이 모두 없다. 시뮬레이터가 자세를 **정확히 알고** 있고,
  가림 여부도 우리가 지정했으며, 필요하면 얼마든지 더 만들 수 있다.

자세 판정 근거
  place_persons.py 의 배치 순서(서있음 4 → 가림쓰러짐 3 → 쓰러짐 6 → 보정 2)를
  seg_color_map.json 의 instance_index 정렬과 맞춰 인스턴스별 자세를 복원한다.
  dataset_stats.py 가 쓰는 것과 같은 규약이며, 그 결과가
  dataset_stats.json 의 instance_counts_by_pose 로 이미 검증되어 있다.

  standing        → person (정상)
  lying           → fallen (쓰러짐)
  lying_occluded  → fallen (가려진 쓰러짐 — 실사의 Hiding(Laying) 에 대응)
  calib           → 제외 (크기 보정용 표본이라 자세 학습 대상이 아님)

크롭 규칙은 실사(make_pose_crops.py)와 **동일**하다. 그래야 섞어 학습할 수 있다.

실행: python make_synth_pose_crops.py [--size 128] [--context 1.6]
출력: data/pose_archive/cls_synth/train/{person,fallen}/*.jpg
"""
import argparse
import json
import shutil
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
RAW = BASE_DIR / "data" / "det" / "synth" / "raw"
CMAP = BASE_DIR / "configs" / "seg_color_map.json"
OUT = BASE_DIR / "data" / "pose_archive" / "cls_synth"

# place_persons.py 기본값 (dataset_stats.py 와 동일해야 한다)
N_STANDING, N_OCCLUDED, N_LYING, N_CALIB = 4, 3, 9, 2

POSE_TO_CLASS = {"standing": "person", "lying": "fallen",
                 "lying_occluded": "fallen", "calib": None, "unknown": None}


def pose_labels(n):
    seq = (["standing"] * N_STANDING
           + ["lying_occluded"] * N_OCCLUDED
           + ["lying"] * (N_LYING - N_OCCLUDED)
           + ["calib"] * N_CALIB)
    if len(seq) < n:
        seq += ["unknown"] * (n - len(seq))
    return seq[:n]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"인코딩 실패: {p}")
    buf.tofile(str(p))


def mask_for_color(seg, rgb):
    """seg 이미지에서 해당 인스턴스 색만 남긴 마스크.

    seg_color_map.json 의 rgb 는 **seg 이미지 바이트 순서와 동일**하다고 명시되어 있다
    (colormap.npy 값 그대로). cv2 가 BGR 로 읽는다고 뒤집으면 다른 인스턴스 색과
    우연히 겹쳐 엉뚱한 개체를 잡는다 — 실제로 그렇게 잘못 뽑은 이력이 있다.
    dataset_stats.py 와 같은 방식(그대로 비교)을 쓴다."""
    c = np.array(rgb, dtype=np.uint8)
    return (cv2.inRange(seg, c, c) > 0).astype(np.uint8)


def bbox_from_mask(mask, min_area=100):
    """가장 큰 연결 성분의 경계 상자. 마네킹이 잔해로 쪼개져 보일 수 있어
    가장 큰 덩어리만 쓰지 않고 전체 마스크의 외곽을 잡는다."""
    if mask.sum() < min_area:
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_square(img, x1, y1, x2, y2, context, size):
    """make_pose_crops.py 와 동일한 규칙 — 정사각형 창 + 가장자리 복제"""
    H, W = img.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * context
    ax0, ay0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    ax1, ay1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    px0, py0 = max(0, -ax0), max(0, -ay0)
    px1, py1 = max(0, ax1 - W), max(0, ay1 - H)
    crop = img[max(ay0, 0):min(ay1, H), max(ax0, 0):min(ax1, W)]
    if crop.size == 0:
        return None
    if px0 or py0 or px1 or py1:
        crop = cv2.copyMakeBorder(crop, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--context", type=float, default=1.6)
    ap.add_argument("--min-side", type=int, default=12,
                    help="이보다 작은 상자는 버린다. 너무 작으면 자세 단서가 없다")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    if not RAW.is_dir():
        raise SystemExit(f"합성 원본이 없습니다: {RAW}")
    cmap = json.loads(CMAP.read_text(encoding="utf-8"))
    by_idx = sorted(cmap["person"], key=lambda e: e["instance_index"])
    poses = pose_labels(len(by_idx))
    entries = [(e["name"], tuple(e["rgb"]), poses[i]) for i, e in enumerate(by_idx)]

    print("인스턴스별 자세 배정")
    for name, rgb, pose in entries:
        print(f"  {name:<26} {pose:<15} → {POSE_TO_CLASS[pose] or '제외'}")

    if out.exists():
        shutil.rmtree(out)
    for c in ("person", "fallen"):
        (out / "train" / c).mkdir(parents=True, exist_ok=True)

    stat, skipped = Counter(), Counter()
    rgb_files = sorted(RAW.glob("rgb_*.png"))
    for n, rgb_path in enumerate(rgb_files):
        idx = rgb_path.stem.split("_")[1]
        seg_path = RAW / f"seg_{idx}.png"
        if not seg_path.exists():
            skipped["seg 없음"] += 1
            continue
        img, seg = imread_u(rgb_path), imread_u(seg_path)
        if img is None or seg is None:
            skipped["읽기 실패"] += 1
            continue

        for name, rgb, pose in entries:
            cls = POSE_TO_CLASS[pose]
            if cls is None:
                continue
            box = bbox_from_mask(mask_for_color(seg, rgb))
            if box is None:
                continue                      # 이 프레임에 안 보이는 인스턴스
            x1, y1, x2, y2 = box
            if max(x2 - x1, y2 - y1) < args.min_side:
                skipped["너무 작음"] += 1
                continue
            patch = crop_square(img, x1, y1, x2, y2, args.context, args.size)
            if patch is None:
                skipped["빈 크롭"] += 1
                continue
            imwrite_u(out / "train" / cls / f"synth_{idx}_{name}.jpg", patch)
            stat[cls] += 1
            stat[f"pose:{pose}"] += 1
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(rgb_files)} 처리")

    tot = stat["person"] + stat["fallen"]
    print("\n=== 합성 자세 크롭 ===")
    print(f"  총 {tot}장 | person {stat['person']} ({stat['person']/max(tot,1)*100:.1f}%) | "
          f"fallen {stat['fallen']} ({stat['fallen']/max(tot,1)*100:.1f}%)")
    print("  자세 원본:", {k[5:]: v for k, v in stat.items() if k.startswith("pose:")})
    if skipped:
        print("  제외:", dict(skipped))
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
