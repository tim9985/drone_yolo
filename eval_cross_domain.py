"""
eval_cross_domain.py — 다른 도메인 정답셋으로 자세 모델을 검증한다

왜 필요한가
  SARD-2 로 학습한 6클래스 자세 모델이 laying_down 재현율 0.972 를 냈다.
  그러나 그것은 **같은 데이터셋의 검증셋** 기준이다. 촬영 장비·계절·지형이
  같으면 성능이 부풀려진다. 실기체에서 쓰려면 **다른 도메인에서도 되는지**를
  봐야 한다.

  문제는 클래스 체계가 다르다는 것이다.
      SARD-2   6클래스 (Running/Walking/laying_down/not_defined/seated/stands)
      NOMAD·Okutama  2클래스 (0=person, 1=fallen)
  그래서 ultralytics 의 val() 을 바로 못 쓴다. 예측을 2클래스로 접어서
  정답과 맞춰야 한다.

매핑
  laying_down                        → fallen(1)
  Walking · stands · seated · Running → person(0)
  not_defined                        → **평가에서 제외**
      자세를 알 수 없다고 라벨된 것이라 정답으로 칠 수도 오답으로 칠 수도 없다.
      실제 운용에서도 이 출력은 '판단 보류 → 스냅샷 전송' 으로 처리한다.

판정 방식
  탐지와 정답을 IoU 로 짝지어(그리디, IoU>=0.3) 클래스가 맞는지 센다.
  mAP 가 아니라 **혼동 행렬과 클래스별 재현율**을 본다 — 우리가 알고 싶은 것은
  "쓰러진 사람을 쓰러졌다고 판정하는가" 하나이기 때문이다.

실행:
  python eval_cross_domain.py --weights runs_person/sard2_pose6/weights/best.pt \
      --data data/det/okutama_pose/images/val
"""
import argparse
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
# SARD-2 클래스 → 2클래스. None 은 평가 제외
SARD_TO_BIN = {0: 0, 1: 0, 2: 1, 3: None, 4: 0, 5: 0}
BIN_NAMES = ["person", "fallen"]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def read_labels(path, W, H):
    out = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        q = ln.split()
        if len(q) < 5:
            continue
        c = int(q[0])
        x, y, w, h = (float(v) for v in q[1:5])
        out.append((c, ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)))
    return out


def main():
    ap = argparse.ArgumentParser(description="다른 도메인 정답셋으로 자세 모델 검증")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="이미지 폴더 (labels 는 자동 탐색)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou-match", type=float, default=0.3,
                    help="탐지와 정답을 같은 대상으로 볼 IoU 하한")
    ap.add_argument("--six-class", action="store_true",
                    help="정답도 6클래스인 경우 (매핑 없이 그대로 비교)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    idir = Path(args.data)
    imgs = sorted(list(idir.glob("*.jpg")) + list(idir.glob("*.png")))
    if not imgs:
        raise SystemExit(f"이미지가 없습니다: {idir}")
    print(f"평가 대상 {len(imgs)}장 · {idir}")

    conf_mat = Counter()      # (정답, 예측) → 개수
    gt_count = Counter()
    n_skip_pred = 0
    n_unmatched_gt = 0
    n_unmatched_pred = 0

    for n, ip in enumerate(imgs, 1):
        lp = ip.parent.parent.parent / "labels" / ip.parent.name / (ip.stem + ".txt")
        if not lp.exists():
            lp = ip.parent.parent / "labels" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        img = imread_u(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = read_labels(lp, W, H)
        if not gts:
            continue

        r = model(img, verbose=False, imgsz=args.imgsz, conf=args.conf)[0]
        preds = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            m = c if args.six_class else SARD_TO_BIN.get(int(c))
            if m is None:
                n_skip_pred += 1
                continue
            preds.append((m, tuple(map(float, b))))

        used = set()
        for gc, gb in gts:
            gt_count[gc] += 1
            best, bi = 0.0, None
            for i, (pc, pb) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= args.iou_match:
                used.add(bi)
                conf_mat[(gc, preds[bi][0])] += 1
            else:
                n_unmatched_gt += 1
                conf_mat[(gc, -1)] += 1        # 탐지 못 함
        n_unmatched_pred += len(preds) - len(used)
        if n % 200 == 0:
            print(f"  {n}/{len(imgs)}")

    names = BIN_NAMES if not args.six_class else [str(i) for i in range(6)]
    K = len(names)
    print(f"\n=== 혼동 행렬 (행 = 정답, 열 = 예측) ===")
    hdr = "".join(f"{names[j][:9]:>11}" for j in range(K))
    print(f"{'':<14}{hdr}{'미탐지':>11}")
    for i in range(K):
        row = "".join(f"{conf_mat[(i, j)]:>11,}" for j in range(K))
        print(f"  {names[i]:<12}{row}{conf_mat[(i, -1)]:>11,}")

    print(f"\n=== 클래스별 성능 ===")
    print(f"{'클래스':<14}{'정답수':>8}{'맞음':>8}{'재현율':>9}{'정밀도':>9}")
    for i in range(K):
        tp = conf_mat[(i, i)]
        tot = gt_count[i]
        pred_i = sum(conf_mat[(j, i)] for j in range(K))
        rec = tp / tot if tot else float("nan")
        prec = tp / pred_i if pred_i else float("nan")
        print(f"  {names[i]:<12}{tot:>8,}{tp:>8,}{rec:>9.3f}{prec:>9.3f}")

    print(f"\n  정답에 못 붙은 탐지 {n_unmatched_pred:,}건 · 놓친 정답 {n_unmatched_gt:,}건")
    if n_skip_pred:
        print(f"  not_defined 로 예측해 평가에서 뺀 탐지 {n_skip_pred:,}건")


if __name__ == "__main__":
    main()
