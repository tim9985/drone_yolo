"""
eval_person.py — 학습 전/후 모델 비교 (NOMAD 실데이터 검증셋 기준)

비교 대상 (있는 것만 자동 선택)
  · weights/yolov8s_visdrone.pt        학습 전 기준선
  · weights/yolov8s_stage1_nomad.pt    NOMAD 파인튜닝
  · weights/yolov8s_stage2_mixed.pt    합성 혼합

측정
  1) NOMAD val 전체 mAP50 / mAP50-95 / precision / recall
  2) **가시성 등급별 재현율** — 가림이 심한 표본에서 얼마나 찾는지가 재난 탐색의 핵심.
     annotations.json 의 visibility 를 crop 파일명으로 역추적해 구간별로 나눈다.
  3) 합성 데이터(쓰러진 자세)에서의 재현율 — 합성 혼합의 효용을 보는 지표

실행: python eval_person.py
출력: eval_person.csv, 콘솔 표
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

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
# 공정 비교: 두 모델 모두 학습에 쓰지 않은 배우(004·008·014·018) 1198장을 공통 검증셋으로 쓴다.
# (10명 모델은 011~020 을 아예 본 적이 없고, 20명 모델은 이들을 val 로 뒀다)
NOMAD_YAML = BASE_DIR / "configs" / "data_nomad20.yaml"
NOMAD_VAL_DIRS = [BASE_DIR / "data" / "dataset_nomad" / "images" / "val",
                  BASE_DIR / "data" / "dataset_nomad_a11_20" / "images" / "val"]
SYNTH_VAL_IMG = BASE_DIR / "data" / "dataset_synth" / "yolo" / "images" / "val"
ANN_JSON = BASE_DIR / "data" / "NOMAD" / "annotations.json"
OUT_CSV = BASE_DIR / "metrics" / "eval_person.csv"

CANDIDATES = [
    ("visdrone_baseline", BASE_DIR / "weights" / "yolov8s_visdrone.pt"),
    ("stage1_nomad10", BASE_DIR / "weights" / "yolov8s_stage1_nomad.pt"),
    ("stage1_nomad20", BASE_DIR / "weights" / "yolov8s_stage1_nomad20.pt"),
    ("stage2_mixed", BASE_DIR / "weights" / "yolov8s_stage2_mixed.pt"),
]
PERSON_NAMES = {"person", "pedestrian", "people"}
IOU_HIT = 0.3     # 재현율 판정용 IoU (소형 객체라 관대하게)


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


def activity_index():
    """activityLabels.json → (actor, distance, frame) 로 활동명을 조회하는 함수 반환.
    Walking / Hiding / Laying / Hiding (Laying) 구간이 프레임 범위로 들어있다."""
    p = BASE_DIR / "data" / "NOMAD" / "activityLabels.json"
    if not p.exists():
        return lambda a, d, f: "unknown"
    tbl = {int(r["id"]): r["labels"] for r in json.load(open(p, encoding="utf-8"))}

    def lookup(actor, dist, frame):
        lab = tbl.get(int(actor), {}).get(str(dist))
        if not lab:
            return "unknown"
        for act, rngs in lab.items():
            for s, e in rngs:
                if int(s) <= frame <= int(e):
                    return act
        return "unlabeled"
    return lookup


def visibility_index():
    """crop 파일명 → 원본 프레임의 visibility. crop 이름은 <원본stem>_cN.jpg."""
    if not ANN_JSON.exists():
        return {}
    idx = {}
    for r in json.load(open(ANN_JSON, encoding="utf-8")):
        if r["annotations"]:
            vs = [int(b.get("visibility", 100)) for b in r["annotations"]]
            idx[r["file_name"].replace(".jpg", "")] = min(vs)
    return idx


def recall_by_group(model, img_dirs, groups_fn, imgsz):
    """이미지별 GT 대비 탐지 재현율을 그룹별로 집계. img_dirs 는 경로 또는 경로 리스트."""
    if isinstance(img_dirs, (str, Path)):
        img_dirs = [Path(img_dirs)]
    files = []
    for d in img_dirs:
        files.extend(sorted(Path(d).glob("*.jpg")))
    hits, total = defaultdict(int), defaultdict(int)
    for img_p in files:
        lab_p = Path(str(img_p).replace("images", "labels")).with_suffix(".txt")
        img = imread_u(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(lab_p, w, h)
        if not gts:
            continue
        res = model(img, verbose=False, imgsz=imgsz)[0]
        preds = []
        for b in res.boxes:
            nm = model.names[int(b.cls[0])]
            if nm not in PERSON_NAMES:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            preds.append([x1, y1, x2 - x1, y2 - y1])
        g = groups_fn(img_p.stem)
        for gt in gts:
            total[g] += 1
            if any(iou(gt, p) >= IOU_HIT for p in preds):
                hits[g] += 1
    return hits, total


def main():
    from ultralytics import YOLO

    models = [(n, p) for n, p in CANDIDATES if p.exists()]
    if not models:
        raise SystemExit("비교할 가중치가 없습니다. 먼저 train_person.py 를 실행하세요.")
    print("비교 대상:", ", ".join(n for n, _ in models))

    vis_idx = visibility_index()
    act_lookup = activity_index()

    def vis_group(stem):
        base = re.sub(r"_c\d+$", "", stem)
        v = vis_idx.get(base)
        if v is None:
            return "unknown"
        if v >= 80:
            return "가림적음(80-100)"
        if v >= 40:
            return "중간(40-70)"
        return "가림심함(10-30)"

    def act_group(stem):
        m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", stem)
        if not m:
            return "unknown"
        return act_lookup(m.group(1), int(m.group(2)), int(m.group(3)))

    rows = []
    for name, wpath in models:
        print(f"\n=== {name} ===")
        model = YOLO(str(wpath))

        # 1) 표준 지표
        m = model.val(data=str(NOMAD_YAML), imgsz=960, verbose=False, plots=False)
        box = m.box
        rec = {"model": name, "mAP50": round(float(box.map50), 4),
               "mAP50_95": round(float(box.map), 4),
               "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4)}
        print(f"  NOMAD val  mAP50 {rec['mAP50']:.3f} | mAP50-95 {rec['mAP50_95']:.3f} "
              f"| P {rec['precision']:.3f} | R {rec['recall']:.3f}")

        # 2) 가시성 등급별 재현율
        hits, total = recall_by_group(model, NOMAD_VAL_DIRS, vis_group, 960)
        for g in sorted(total):
            r = hits[g] / total[g]
            rec[f"recall_{g}"] = round(r, 4)
            print(f"    {g:16} 재현율 {r:.3f}  ({hits[g]}/{total[g]})")

        # 2-b) 활동별 재현율 — 쓰러진 자세(Laying 계열)가 핵심 지표
        ah, at = recall_by_group(model, NOMAD_VAL_DIRS, act_group, 960)
        for g in sorted(at):
            r = ah[g] / at[g]
            rec[f"recall_act_{g}"] = round(r, 4)
            mark = "  ← 쓰러짐" if "Laying" in g else ""
            print(f"    [활동] {g:18} 재현율 {r:.3f}  ({ah[g]}/{at[g]}){mark}")

        # 3) 합성(쓰러진 자세) 재현율
        if SYNTH_VAL_IMG.exists():
            h2, t2 = recall_by_group(model, SYNTH_VAL_IMG, lambda s: "synth", 960)
            if t2["synth"]:
                r = h2["synth"] / t2["synth"]
                rec["recall_synth_lying"] = round(r, 4)
                print(f"    {'합성(쓰러짐)':16} 재현율 {r:.3f}  ({h2['synth']}/{t2['synth']})")
        rows.append(rec)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
