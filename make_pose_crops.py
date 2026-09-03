"""
make_pose_crops.py — 자세 분류기용 크롭 데이터셋 생성

왜 크롭인가
  탐지기를 2클래스로 바꾸려다 실패했다(pose2: 쓰러짐 재현율 0.153).
  기본 학습률이 통합 모델의 특징을 망가뜨렸고, 박스 손실까지 1.2 → 1.9 로 나빠졌다.
  탐지기는 건드리지 않고, **탐지된 상자만 잘라 별도 분류기**에 넘기는 2단계로 간다.

    stage1_all 로 사람 탐지 (무손상)
        → 상자를 잘라냄
        → 소형 분류기: 정상 / 쓰러짐

  분류기는 1280x720 전체가 아니라 100px 안팎 패치만 보므로 학습이 수십 배 싸다.

크롭 규칙
  상자 중심을 기준으로 **정사각형**을 잘라낸다. 한 변은 긴 변의 CONTEXT 배.
  · 정사각형: 누운 자세는 가로로, 선 자세는 세로로 길다. 비율을 유지한 채
    정사각형에 넣어야 **종횡비 자체가 분류 단서로 남는다.**
  · 여백(CONTEXT): 주변 지면이 조금 보여야 "누워 있다"가 판단된다.
    상자에 딱 맞추면 배경 문맥이 사라진다.

출력 (ultralytics 분류 규약)
  data/pose_archive/cls/{train,val}/{person,fallen}/*.jpg

실행: python make_pose_crops.py [--size 128] [--context 1.6]
"""
import argparse
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
SRC = BASE_DIR / "data" / "pose_archive" / "det2"
OUT = BASE_DIR / "data" / "pose_archive" / "cls"
CLASS_NAMES = {0: "person", 1: "fallen"}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"인코딩 실패: {p}")
    buf.tofile(str(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128, help="출력 패치 한 변 (px)")
    ap.add_argument("--context", type=float, default=1.6,
                    help="상자 긴 변의 몇 배를 잘라낼지. 1.0 이면 여백 없음")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        for name in CLASS_NAMES.values():
            (out / sp / name).mkdir(parents=True, exist_ok=True)

    stat = {sp: Counter() for sp in ("train", "val")}
    skipped = Counter()

    for sp in ("train", "val"):
        img_dir = SRC / "images" / sp
        if not img_dir.is_dir():
            continue
        for img_p in sorted(img_dir.glob("*.jpg")):
            lab_p = SRC / "labels" / sp / f"{img_p.stem}.txt"
            if not lab_p.exists():
                skipped["라벨 없음"] += 1
                continue
            img = imread_u(img_p)
            if img is None:
                skipped["읽기 실패"] += 1
                continue
            H, W = img.shape[:2]

            for i, ln in enumerate(lab_p.read_text(encoding="utf-8").splitlines()):
                v = ln.split()
                if len(v) != 5:
                    continue
                cls = int(v[0])
                cx, cy, bw, bh = [float(x) for x in v[1:]]
                cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H

                # 정사각형 창 — 종횡비를 살리려면 창이 정사각형이어야 한다
                side = max(bw, bh) * args.context
                x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
                x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))

                # 화면 밖은 잘라내고, 모자란 만큼은 가장자리 복제로 채운다.
                # 검은 여백을 넣으면 분류기가 그 패턴을 단서로 삼는다.
                px0, py0 = max(0, -x0), max(0, -y0)
                px1, py1 = max(0, x1 - W), max(0, y1 - H)
                crop = img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
                if crop.size == 0:
                    skipped["빈 크롭"] += 1
                    continue
                if px0 or py0 or px1 or py1:
                    crop = cv2.copyMakeBorder(crop, py0, py1, px0, px1, cv2.BORDER_REPLICATE)

                patch = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
                name = CLASS_NAMES[cls]
                imwrite_u(out / sp / name / f"{img_p.stem}_{i}.jpg", patch)
                stat[sp][name] += 1

    print("=== 자세 분류용 크롭 ===")
    print(f"  패치 {args.size}x{args.size} · 여백 배율 {args.context}")
    for sp in ("train", "val"):
        c = stat[sp]
        tot = sum(c.values())
        if not tot:
            continue
        print(f"  {sp:5} 총 {tot:>5}장 | " +
              " | ".join(f"{k} {c[k]:>5} ({c[k]/tot*100:4.1f}%)" for k in CLASS_NAMES.values()))
    if skipped:
        print("  제외:", dict(skipped))
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
