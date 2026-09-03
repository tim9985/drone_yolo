"""
eval_domain.py — 도메인별 분리 평가

왜 나눠서 보나
  통합 학습 모델의 mAP 0.850 은 **통합 검증셋 2,769장** 기준이고,
  이전 모델들의 0.643 은 **NOMAD 검증셋 1,198장** 기준이다. 잣대가 다르므로
  그대로 비교하면 안 된다. 도메인을 분리해야 아래 두 가지를 알 수 있다.

    (a) 겨울 데이터를 넣은 대가로 기존(여름·농장) 성능이 깎였는가
    (b) 최종 시현 환경(11~12월 산지)에 가까운 조건에서 실제로 몇 %를 찾는가

도메인
  nomad_summer   NOMAD val 1,198장  — 미국 농장·여름·정오. 이전 모델과 직접 비교 가능
  wisard_sept    WiSARD 9월 289장   — 가을 산지 (FHL)
  wisard_jan     WiSARD 1월 1,282장 — 겨울·설경 (Baker). **시현 환경에 가장 가까움**
  combined       전체 2,769장       — 학습 중 표시되던 값과 대조용

실행: python eval_domain.py
출력: eval_domain.csv + 콘솔 표
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "metrics" / "_lists"
OUT_CSV = BASE_DIR / "metrics" / "eval_domain.csv"
IMGSZ = 960

NOMAD_VAL = [BASE_DIR / "data" / "det" / "nomad_actor01_10" / "images" / "val",
             BASE_DIR / "data" / "det" / "nomad_actor11_20" / "images" / "val"]
WISARD_VAL = BASE_DIR / "data" / "det" / "wisard" / "images" / "val"
# 1월(겨울) 비행은 DJI_0582 (1-10-2022 ...). 나머지(DJI_0403/0409)는 9월.
JAN_PREFIX = "DJI_0582"

MODELS = [
    ("visdrone_baseline", BASE_DIR / "weights" / "yolov8s_visdrone.pt"),
    ("nomad30", BASE_DIR / "weights" / "yolov8s_stage1_nomad30.pt"),
    ("all_nomad+wisard", BASE_DIR / "runs_person" / "stage1_all" / "weights" / "best.pt"),
]

PERSON_NAMES = {"person", "pedestrian", "people"}
IOU_HIT = 0.3


def build_domains():
    """도메인 이름 → 이미지 경로 리스트"""
    nomad = []
    for d in NOMAD_VAL:
        nomad.extend(sorted(d.glob("*.jpg")))
    wis = sorted(WISARD_VAL.glob("*.jpg"))
    jan = [p for p in wis if p.name.startswith(JAN_PREFIX)]
    sep = [p for p in wis if not p.name.startswith(JAN_PREFIX)]
    return {
        "nomad_summer": nomad,
        "wisard_sept": sep,
        "wisard_jan": jan,
        "combined": nomad + wis,
    }


def write_yaml(name, imgs):
    """ultralytics 는 val 에 이미지 목록 txt 를 받는다.
    라벨 경로는 /images/ → /labels/ 치환으로 자동 유도되므로 폴더 구조를 건드릴 필요가 없다."""
    TMP_DIR.mkdir(exist_ok=True)
    lst = TMP_DIR / f"{name}.txt"
    lst.write_text("\n".join(str(p) for p in imgs), encoding="utf-8")
    y = TMP_DIR / f"{name}.yaml"
    # ultralytics 는 val 만 쓸 때도 train 키를 요구한다. 검증에는 쓰이지 않으므로 같은 목록을 준다.
    y.write_text(f"train: {lst.as_posix()}\nval: {lst.as_posix()}\nnc: 1\nnames: ['person']\n",
                 encoding="utf-8")
    return y


# ── 쓰러진 자세 재현율 (NOMAD 전용) ────────────────────────────────
def imread_u(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_yolo_labels(label_path, w, h):
    out = []
    if not label_path.exists():
        return out
    for ln in label_path.read_text(encoding="utf-8").splitlines():
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
    import json
    p = BASE_DIR / "data" / "raw" / "NOMAD" / "activityLabels.json"
    if not p.exists():
        return None
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


def laying_recall(model, imgs):
    """NOMAD val 에서 활동별 재현율. 쓰러짐(Laying 계열)이 우리 과제의 핵심 지표."""
    look = activity_lookup()
    if look is None:
        return {}
    hits, total = defaultdict(int), defaultdict(int)
    for img_p in imgs:
        m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", img_p.stem)
        if not m:
            continue
        act = look(m.group(1), int(m.group(2)), int(m.group(3)))
        img = imread_u(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(Path(str(img_p).replace("images", "labels")).with_suffix(".txt"), w, h)
        if not gts:
            continue
        res = model(img, verbose=False, imgsz=IMGSZ)[0]
        preds = []
        for b in res.boxes:
            if model.names[int(b.cls[0])] not in PERSON_NAMES:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append([x1, y1, x2 - x1, y2 - y1])
        for gt in gts:
            total[act] += 1
            if any(iou(gt, p) >= IOU_HIT for p in preds):
                hits[act] += 1
    return {a: (hits[a], total[a]) for a in sorted(total)}


def main():
    import argparse
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(
        description="도메인(계절·데이터셋)별로 나눠 탐지 성능을 잰다")
    ap.add_argument("--weights", nargs="*", default=None, metavar="PATH",
                    help="평가할 가중치. 여러 개 가능. 생략하면 MODELS 기본 목록을 쓴다. "
                         "이름은 파일 경로에서 자동으로 만든다")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="추론 해상도. 학습 때 쓴 값과 맞춰야 공정한 비교가 된다")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="신뢰도 임계값 (기본값은 F2 스윕으로 정한 운용값)")
    args = ap.parse_args()

    global MODELS
    if args.weights:
        # 경로에서 이름을 짓는다: runs_person/yolo11s_1280/weights/best.pt → yolo11s_1280
        named = []
        for w in args.weights:
            wp = Path(w).resolve()
            name = wp.parent.parent.name if wp.name in ("best.pt", "last.pt") else wp.stem
            named.append((name, wp))
        MODELS = named

    doms = build_domains()
    print("도메인 구성")
    for k, v in doms.items():
        print(f"  {k:14} {len(v):>5}장")

    models = [(n, p) for n, p in MODELS if p.exists()]
    missing = [n for n, p in MODELS if not p.exists()]
    if missing:
        print(f"  (없어서 건너뜀: {', '.join(missing)})")
    if not models:
        raise SystemExit("평가할 가중치가 없습니다.")

    yamls = {k: write_yaml(k, v) for k, v in doms.items()}
    rows = []

    for mname, wpath in models:
        print(f"\n{'='*62}\n{mname}\n{'='*62}")
        model = YOLO(str(wpath))
        for dname, y in yamls.items():
            r = model.val(data=str(y), imgsz=IMGSZ, verbose=False, plots=False)
            b = r.box
            rec = {"model": mname, "domain": dname, "images": len(doms[dname]),
                   "mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4),
                   "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4)}
            rows.append(rec)
            print(f"  {dname:14} mAP50 {rec['mAP50']:.3f} | mAP50-95 {rec['mAP50_95']:.3f} "
                  f"| P {rec['precision']:.3f} | R {rec['recall']:.3f}")

        # 쓰러진 자세 재현율은 NOMAD 에만 라벨이 있다
        act = laying_recall(model, doms["nomad_summer"])
        for a, (h, t) in act.items():
            mark = "  ← 쓰러짐" if "Laying" in a else ""
            print(f"    [활동] {a:20} 재현율 {h/t:.3f} ({h}/{t}){mark}")
            rows.append({"model": mname, "domain": f"nomad_act::{a}", "images": t,
                         "recall": round(h / t, 4)})

    keys = ["model", "domain", "images", "mAP50", "mAP50_95", "precision", "recall"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
