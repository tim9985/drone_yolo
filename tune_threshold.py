"""
tune_threshold.py — 신뢰도 임계값 조정 (재학습 없음)

문제
  기본 임계값 0.25 에서 통합 모델은 사람 없는 배경 289장에 장당 0.72개를 오탐한다
  (NOMAD 전용 모델의 3.6배). 재난 탐색에서 오탐은 구조대를 헛걸음시킨다.
  반대로 임계값을 올리면 실제 조난자를 놓친다. **둘의 균형점을 찾아야 한다.**

방법
  임계값마다 추론을 다시 돌리면 느리다. 대신 **conf=0.001 로 한 번만 추론**해
  모든 후보 상자를 신뢰도와 함께 모아두고, 임계값을 바꿔가며 오프라인으로 집계한다.
  (2,769장 × 1회 추론 ≈ 3분, 임계값 20개를 다시 돌리면 1시간)

집계
  이미지마다 신뢰도 내림차순으로 GT 와 탐욕적 매칭(IoU ≥ 0.3).
  매칭된 예측 = TP, 남은 예측 = FP, 남은 GT = FN.

평가 대상
  nomad_summer   여름·농장 1,198장
  wisard_jan     겨울·설경 1,282장   ← 시현 환경에 가장 가까움
  laying         위 중 쓰러진 자세만 (NOMAD 활동 라벨)
  background     사람 없는 배경 289장 ← 오탐 전용

실행: python tune_threshold.py
출력: threshold_tuning.csv + 콘솔 표
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR / "runs_person" / "stage1_all" / "weights" / "best.pt"
OUT_CSV = BASE_DIR / "metrics" / "threshold_tuning.csv"
IMGSZ = 960
IOU_HIT = 0.3
SCAN_CONF = 0.001          # 이 값으로 한 번만 추론해 후보를 모두 확보
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
PERSON_NAMES = {"person", "pedestrian", "people"}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_gt(img_p, w, h):
    lp = Path(str(img_p).replace("images", "labels")).with_suffix(".txt")
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = [float(v) for v in p]
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return out


def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def activity_lookup():
    p = BASE_DIR / "data" / "raw" / "NOMAD" / "activityLabels.json"
    if not p.exists():
        return lambda *_: "unknown"
    tbl = {int(r["id"]): r["labels"] for r in json.load(open(p, encoding="utf-8"))}

    def f(actor, dist, frame):
        lab = tbl.get(int(actor), {}).get(str(dist))
        if not lab:
            return "unknown"
        for act, rngs in lab.items():
            for s, e in rngs:
                if int(s) <= frame <= int(e):
                    return act
        return "unlabeled"
    return f


def collect(model, imgs, tag):
    """이미지별 (GT 목록, 예측 목록[(conf, box)]) 을 한 번의 추론으로 모은다."""
    out = []
    for i, p in enumerate(imgs):
        img = imread_u(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model(img, verbose=False, imgsz=IMGSZ, conf=SCAN_CONF)[0]
        preds = []
        for b in r.boxes:
            if model.names[int(b.cls[0])] not in PERSON_NAMES:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append((float(b.conf[0]), [x1, y1, x2 - x1, y2 - y1]))
        preds.sort(key=lambda t: -t[0])
        out.append({"path": p, "gt": load_gt(p, w, h), "pred": preds})
        if (i + 1) % 400 == 0:
            print(f"    {tag} {i+1}/{len(imgs)}")
    return out


def score(records, conf):
    """주어진 임계값에서 TP/FP/FN 집계"""
    tp = fp = fn = 0
    for r in records:
        preds = [b for c, b in r["pred"] if c >= conf]
        gts = list(r["gt"])
        used = [False] * len(gts)
        for pb in preds:                      # 신뢰도 높은 순서로 매칭
            best, bi = 0.0, -1
            for gi, gb in enumerate(gts):
                if used[gi]:
                    continue
                v = iou(pb, gb)
                if v > best:
                    best, bi = v, gi
            if best >= IOU_HIT:
                used[bi] = True
                tp += 1
            else:
                fp += 1
        fn += used.count(False)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    import argparse
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help="평가할 가중치. 모델 간 비교 시 같은 임계값 격자에서 재야 공정하다.")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    # 상대경로로 넘어와도 동작하도록 절대경로로 정규화
    weights, out_csv = Path(args.weights).resolve(), Path(args.out).resolve()
    if not weights.exists():
        raise SystemExit(f"가중치 없음: {weights}")

    nomad = []
    for d in ("det/nomad_actor01_10", "det/nomad_actor11_20"):
        nomad.extend(sorted((BASE_DIR / "data" / d / "images" / "val").glob("*.jpg")))
    wis = sorted((BASE_DIR / "data" / "det" / "wisard" / "images" / "val").glob("*.jpg"))
    jan = [p for p in wis if p.name.startswith("DJI_0582")]
    bg = [p for p in wis if not p.name.startswith("DJI_0582")]

    print(f"가중치: {weights.relative_to(BASE_DIR)}")
    print(f"대상: 여름 {len(nomad)} / 겨울 {len(jan)} / 배경 {len(bg)}\n")

    model = YOLO(str(weights))
    print("  추론 수집 중 (임계값별 재추론 없이 1회만)")
    rec_nomad = collect(model, nomad, "여름")
    rec_jan = collect(model, jan, "겨울")
    rec_bg = collect(model, bg, "배경")

    # 쓰러진 자세만 추린 부분집합
    look = activity_lookup()
    rec_lay = []
    for r in rec_nomad:
        m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", r["path"].stem)
        if m and "Laying" in look(m.group(1), int(m.group(2)), int(m.group(3))):
            rec_lay.append(r)
    print(f"  쓰러진 자세 부분집합 {len(rec_lay)}장\n")

    rows = []
    hdr = (f"{'conf':>5} | {'여름 R':>7} {'여름 P':>7} | {'겨울 R':>7} {'겨울 P':>7} | "
           f"{'쓰러짐 R':>8} | {'배경 오탐/장':>11} {'오탐장%':>8} | {'F1(겨울)':>8}")
    print(hdr)
    print("-" * len(hdr))
    for c in THRESHOLDS:
        tn, fpn, fnn = score(rec_nomad, c)
        tj, fpj, fnj = score(rec_jan, c)
        tl, fpl, fnl = score(rec_lay, c)
        _, fpb, _ = score(rec_bg, c)
        dirty = sum(1 for r in rec_bg if any(cf >= c for cf, _ in r["pred"]))
        pn, rn, _ = prf(tn, fpn, fnn)
        pj, rj, fj = prf(tj, fpj, fnj)
        _, rl, _ = prf(tl, fpl, fnl)
        fpi = fpb / len(rec_bg)
        rows.append({"conf": c, "summer_recall": round(rn, 4), "summer_precision": round(pn, 4),
                     "winter_recall": round(rj, 4), "winter_precision": round(pj, 4),
                     "winter_f1": round(fj, 4), "laying_recall": round(rl, 4),
                     "bg_fp_per_image": round(fpi, 3),
                     "bg_dirty_pct": round(dirty / len(rec_bg) * 100, 1)})
        print(f"{c:>5.2f} | {rn:>7.3f} {pn:>7.3f} | {rj:>7.3f} {pj:>7.3f} | "
              f"{rl:>8.3f} | {fpi:>11.2f} {dirty/len(rec_bg)*100:>7.1f}% | {fj:>8.3f}")

    best_f1 = max(rows, key=lambda r: r["winter_f1"])
    print(f"\n겨울 F1 최대: conf={best_f1['conf']:.2f} (F1 {best_f1['winter_f1']:.3f})")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"저장: {out_csv}")


if __name__ == "__main__":
    main()
