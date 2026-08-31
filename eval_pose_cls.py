"""
eval_pose_cls.py — 자세 분류기 평가

왜 정확도(top1)만 보면 안 되나
  검증셋이 person 989 : fallen 209 로 4.7:1 불균형이다.
  **전부 person 이라 답해도 정확도 82.6% 가 나온다.**
  따라서 쓰러짐 재현율과 정밀도를 따로 본다.

측정
  · 혼동 행렬 (정상/쓰러짐)
  · 클래스별 정밀도·재현율·F1
  · "전부 person" 기준선 대비 실제 이득
  · 신뢰도 임계값별 쓰러짐 재현율 — 탐지기에서와 같은 맞교환이 여기도 있다
  · 2단계 결합 성능 추정 (탐지 재현율 x 분류 재현율)

실행: python eval_pose_cls.py [--weights weights/pose_cls.pt]
출력: metrics/pose_cls_eval.csv + 콘솔 표
"""
import argparse
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
VAL = BASE_DIR / "data" / "dataset_pose_cls" / "val"
DEFAULT_W = BASE_DIR / "weights" / "pose_cls.pt"
OUT_CSV = BASE_DIR / "metrics" / "pose_cls_eval.csv"

# 탐지기(stage1_all, 임계값 0.15)가 쓰러진 사람을 찾아내는 비율.
# 2단계 결합 성능을 추정할 때 쓴다. 근거: metrics/threshold_tuning.csv
DETECTOR_FALLEN_RECALL = 0.608


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_W))
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"가중치 없음: {w}\n먼저 train_pose_cls.py 를 실행하세요.")

    from ultralytics import YOLO
    model = YOLO(str(w))
    names = model.names                       # {0: 'fallen', 1: 'person'} 순서는 폴더명 정렬
    idx_of = {v: k for k, v in names.items()}
    print(f"가중치: {w.name}   클래스: {names}")

    # 실제 라벨은 폴더명이다
    samples = []
    for cls_name in ("person", "fallen"):
        d = VAL / cls_name
        for p in sorted(d.glob("*.jpg")):
            samples.append((p, cls_name))
    print(f"검증 표본 {len(samples)}장 "
          f"(person {sum(1 for _, c in samples if c=='person')} / "
          f"fallen {sum(1 for _, c in samples if c=='fallen')})\n")

    # 한 번만 추론하고 확률을 모아둔다 — 임계값 훑기를 위해
    probs, truth = [], []
    for i, (p, cls_name) in enumerate(samples):
        img = imread_u(p)
        if img is None:
            continue
        r = model(img, verbose=False)[0]
        pr = r.probs.data.tolist()
        probs.append(pr[idx_of["fallen"]])     # 쓰러짐일 확률
        truth.append(1 if cls_name == "fallen" else 0)
        if (i + 1) % 400 == 0:
            print(f"  추론 {i+1}/{len(samples)}")
    probs, truth = np.array(probs), np.array(truth)

    def metrics(th):
        pred = (probs >= th).astype(int)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        tn = int(((pred == 0) & (truth == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(truth)
        return dict(th=th, tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=prec, recall=rec, f1=f1, accuracy=acc)

    base = metrics(0.5)
    print("=== 기본 임계값 0.50 ===")
    print(f"{'':12}{'예측 person':>12}{'예측 fallen':>12}")
    print(f"{'실제 person':12}{base['tn']:>12}{base['fp']:>12}")
    print(f"{'실제 fallen':12}{base['fn']:>12}{base['tp']:>12}")
    print(f"\n쓰러짐 정밀도 {base['precision']:.3f} | 재현율 {base['recall']:.3f} | "
          f"F1 {base['f1']:.3f} | 정확도 {base['accuracy']:.3f}")

    trivial = (truth == 0).mean()
    print(f"\n'전부 person' 기준선 정확도 {trivial:.3f}  "
          f"→ 실제 이득 {base['accuracy']-trivial:+.3f}")

    print("\n=== 임계값별 (쓰러짐 판정 기준) ===")
    print(f"{'th':>6}{'정밀도':>9}{'재현율':>9}{'F1':>8}{'정확도':>9}{'오탐(FP)':>10}")
    rows = []
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = metrics(th)
        rows.append(m)
        print(f"{th:>6.2f}{m['precision']:>9.3f}{m['recall']:>9.3f}"
              f"{m['f1']:>8.3f}{m['accuracy']:>9.3f}{m['fp']:>10}")

    best_f1 = max(rows, key=lambda m: m["f1"])
    print(f"\nF1 최대: 임계값 {best_f1['th']:.2f} "
          f"(정밀도 {best_f1['precision']:.3f} / 재현율 {best_f1['recall']:.3f})")

    print("\n=== 2단계 결합 성능 추정 ===")
    print(f"탐지기(stage1_all, conf 0.15)의 쓰러짐 재현율 {DETECTOR_FALLEN_RECALL:.3f}")
    print(f"{'분류 임계값':>10}{'분류 재현율':>12}{'결합 재현율':>12}")
    for m in rows:
        if m["th"] in (0.2, 0.3, 0.5, 0.7):
            print(f"{m['th']:>10.2f}{m['recall']:>12.3f}"
                  f"{DETECTOR_FALLEN_RECALL*m['recall']:>12.3f}")
    print("\n※ 탐지 단계에서 놓친 대상은 분류기가 손댈 수 없다.")
    print("   결합 재현율 = 탐지 재현율 x 분류 재현율 이 상한이다.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
