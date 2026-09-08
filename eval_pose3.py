"""
eval_pose3.py — 자세 3클래스 모델의 합격 기준 4개를 한 번에 잰다

왜 이 스크립트가 따로 필요한가
  자세를 얻으려다 탐지를 잃으면 손해다. 그래서 두 축을 **동시에** 봐야 한다.

    (A) 탐지 강건성 — 클래스 무시하고 사람을 찾기라도 하는가
    (B) 자세 능력   — 찾은 다음 쓰러짐을 맞히는가

  기존 도구로는 이게 안 됐다.
    · eval_domain.py 는 val 야믈을 nc:1 로 써서 다중 클래스 모델에 못 물린다
    · eval_cross_domain.py 는 SARD 6클래스 → 2클래스 매핑이 하드코딩돼 있다

  게다가 비교 대상(stage1_all 1클래스, sard2_pose6 6클래스)은 클래스 수가
  제각각이다. **같은 코드로 다시 재지 않으면 공정한 비교가 아니다.**
  그래서 모델 클래스 수를 보고 매핑을 자동으로 고른다.

합격 기준 (문서 06)
  NOMAD  사람 발견율  >= 0.60   기존 강건성 회복 (stage1_all 원본 0.667)
  WiSARD 사람 발견율  >= 0.85   기존 강건성 회복 (stage1_all 원본 0.884)
  SARD   쓰러짐 재현율 >= 0.90   자세 능력 유지
  NOMAD  쓰러짐 재현율 >= 0.50   도메인 이전 확보

  앞의 둘이 우선이다.

판정 방식
  탐지와 정답을 그리디 IoU 매칭(기본 0.3)으로 짝짓는다. mAP 가 아니라
  **재현율과 정밀도**를 본다 — 알고 싶은 것이 "놓쳤는가"뿐이기 때문이다.
  재난 탐색에서 미탐지는 정탐지 실패보다 훨씬 비싸다.

실행
  python eval_pose3.py --weights runs_person/pose3_sn_freeze/weights/best.pt
  python eval_pose3.py --weights weights/yolov8s_stage1_all.pt \
                                 weights/yolov8s_sard2_pose6.pt \
                                 runs_person/pose3_sn_freeze/weights/best.pt
출력: metrics/eval_pose3.csv + 콘솔 표
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_CSV = BASE_DIR / "metrics" / "eval_pose3.csv"

# 평가 공통 축: 0 person / 1 fallen / None 은 평가 제외
# ambiguous 는 정답으로도 오답으로도 셀 수 없다. 운용에서도 '판단 보류'다.
PERSON, FALLEN = 0, 1

# 모델 클래스 수 → (평가축 매핑, 설명)
#   1클래스 stage1_all       : 자세를 모르므로 전부 person 취급 (탐지 축만 유효)
#   3클래스 pose3            : person/fallen/ambiguous
#   6클래스 sard2_pose6      : SARD 원본 순서
PRED_MAPS = {
    1: ({0: PERSON}, "1클래스(person) — 탐지 축만 유효"),
    3: ({0: PERSON, 1: FALLEN, 2: None}, "3클래스(person/fallen/ambiguous)"),
    6: ({0: PERSON, 1: PERSON, 2: FALLEN, 3: None, 4: PERSON, 5: PERSON},
        "6클래스 SARD 원본"),
}
# SARD 원본 6클래스 정답 → 평가축
SARD_GT_MAP = {0: PERSON, 1: PERSON, 2: FALLEN, 3: None, 4: PERSON, 5: PERSON}


def imread_u(p):
    import cv2
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


def label_path_for(img):
    """images/<split>/x.jpg → labels/<split>/x.txt. 두 가지 배치 모두 지원."""
    for cand in (img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt"),
                 img.parent.parent / "labels" / (img.stem + ".txt")):
        if cand.exists():
            return cand
    return None


def read_gt(path, W, H, gt_map):
    out = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        c = gt_map(int(v[0]))
        x, y, w, h = (float(t) for t in v[1:5])
        out.append((c, ((x - w / 2) * W, (y - h / 2) * H,
                        (x + w / 2) * W, (y + h / 2) * H)))
    return out


def build_domains(limit=None):
    """평가 도메인 → (이미지 목록, 정답 매핑, 자세 정답이 있는가)"""
    D = BASE_DIR / "data"

    nomad_det = []
    for s in ("nomad_actor01_10", "nomad_actor11_20"):
        nomad_det += sorted((D / "det" / s / "images" / "val").glob("*.jpg"))

    wisard = sorted((D / "det" / "wisard" / "images" / "val").glob("*.jpg"))

    sard_test = sorted((D / "raw" / "sard2" / "search-and-rescue-2" / "test"
                        / "images").glob("*.jpg"))

    # 3클래스 검증셋에서 출처별로 갈라 쓴다 (파일명 접두어)
    pose_val = sorted((D / "det" / "pose3_sn" / "images" / "val").glob("*.jpg"))
    nomad_pose = [p for p in pose_val if p.name.startswith("nomad_")]
    sard_pose = [p for p in pose_val if p.name.startswith("sard_")]

    # Okutama — 학습에 전혀 쓰지 않은 **실제 드론 영상**. 정답이 2클래스
    # (0 person / 1 fallen) 로 붙어 있어 자세까지 정량 검증할 수 있는 유일한 외부 도메인이다.
    okutama = sorted((D / "det" / "okutama_pose" / "images" / "val").glob("*.jpg"))

    doms = {
        # (A) 탐지 강건성 — 정답이 1클래스 person
        "NOMAD 탐지":  (nomad_det, lambda c: PERSON, False),
        "WiSARD 탐지": (wisard,    lambda c: PERSON, False),
        # (C) 미학습 외부 도메인 — 탐지와 자세를 함께 본다
        "Okutama(미학습)": (okutama, lambda c: {0: PERSON, 1: FALLEN}.get(c), True),
        # (B) 자세 능력 — 정답에 자세가 있다
        "SARD 자세(test)":  (sard_test,  lambda c: SARD_GT_MAP.get(c), True),
        "SARD 자세(val)":   (sard_pose,  lambda c: {0: PERSON, 1: FALLEN, 2: None}.get(c), True),
        "NOMAD 자세(val)":  (nomad_pose, lambda c: {0: PERSON, 1: FALLEN, 2: None}.get(c), True),
    }
    if limit:
        # 앞에서 자르면 안 된다. WiSARD val 은 파일명 정렬 시 **앞쪽 291장이 전부
        # 음성 표본**(사람 없음)이라 --limit 60 을 주면 평가할 정답이 하나도 안 남는다.
        # 라벨이 있는 것만 남기고 균등 간격으로 뽑는다.
        def sample(imgs):
            pos = [p for p in imgs if (lp := label_path_for(p)) is not None
                   and lp.read_text(encoding="utf-8", errors="ignore").strip()]
            if len(pos) <= limit:
                return pos
            step = len(pos) / limit
            return [pos[int(i * step)] for i in range(limit)]
        doms = {k: (sample(v[0]), v[1], v[2]) for k, v in doms.items()}
    return doms


def evaluate(model, pred_map, imgs, gt_map, imgsz, conf, iou_th, device=None):
    """한 도메인 평가. (탐지 통계, 자세 혼동행렬) 반환"""
    n_gt = Counter()          # 평가축 클래스별 정답 수
    n_hit_any = Counter()     # 클래스 무시하고 매칭된 수 (탐지 축)
    conf_mat = Counter()      # (정답, 예측) — 자세 축
    n_pred_unmatched = 0
    n_pred_total = 0

    for ip in imgs:
        lp = label_path_for(ip)
        if lp is None:
            continue
        img = imread_u(ip)
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = read_gt(lp, W, H, gt_map)
        if not gts:
            continue

        r = model(img, verbose=False, imgsz=imgsz, conf=conf, device=device)[0]
        preds = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                        r.boxes.cls.cpu().numpy().astype(int)):
            n_pred_total += 1
            preds.append((pred_map.get(int(c), None), tuple(map(float, b))))

        used = set()
        for gc, gb in gts:
            n_gt[gc] += 1
            best, bi = 0.0, None
            for i, (_pc, pb) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= iou_th:
                used.add(bi)
                n_hit_any[gc] += 1                      # 탐지 축: 클래스 무시
                conf_mat[(gc, preds[bi][0])] += 1       # 자세 축: 클래스 비교
            else:
                conf_mat[(gc, "miss")] += 1
        n_pred_unmatched += len(preds) - len(used)

    return dict(n_gt=n_gt, n_hit_any=n_hit_any, conf_mat=conf_mat,
                n_pred_unmatched=n_pred_unmatched, n_pred_total=n_pred_total)


def main():
    ap = argparse.ArgumentParser(description="자세 3클래스 모델 합격 기준 평가")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15,
                    help="쓰러진 자세는 임계값에 특히 민감하다 (0.25→0.15 로 재현율 0.531→0.608)")
    ap.add_argument("--iou-match", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None, help="도메인당 이미지 수 제한 (빠른 점검용)")
    ap.add_argument("--device", default=None,
                    help="cpu 로 두면 학습이 GPU 를 쓰는 중에도 나란히 돌릴 수 있다")
    args = ap.parse_args()

    from ultralytics import YOLO

    doms = build_domains(args.limit)
    for name, (imgs, _g, _p) in doms.items():
        print(f"  {name:<18} {len(imgs):>6,}장")
    print()

    rows = []
    for wp in args.weights:
        wpath = Path(wp)
        if not wpath.exists():
            print(f"건너뜀 — 가중치 없음: {wpath}")
            continue
        model = YOLO(str(wpath))
        nc = len(model.names)
        if nc not in PRED_MAPS:
            print(f"건너뜀 — 지원하지 않는 클래스 수 {nc}: {wpath}")
            continue
        pred_map, desc = PRED_MAPS[nc]
        tag = wpath.stem if wpath.stem != "best" else wpath.parent.parent.name
        print(f"=== {tag} · {desc} ===")

        for dname, (imgs, gt_map, has_pose) in doms.items():
            if not imgs:
                continue
            r = evaluate(model, pred_map, imgs, gt_map, args.imgsz,
                         args.conf, args.iou_match, args.device)
            tot_gt = sum(r["n_gt"].values())
            if not tot_gt:
                continue
            # 탐지 축 — 클래스 무시한 발견율
            det_rec = sum(r["n_hit_any"].values()) / tot_gt
            det_prec = (sum(r["n_hit_any"].values()) / r["n_pred_total"]
                        if r["n_pred_total"] else float("nan"))
            # 자세 축 — fallen 재현율/정밀도
            f_gt = r["n_gt"][FALLEN]
            f_tp = r["conf_mat"][(FALLEN, FALLEN)]
            f_pred = sum(v for (g, p), v in r["conf_mat"].items() if p == FALLEN)
            f_rec = f_tp / f_gt if f_gt else float("nan")
            f_prec = f_tp / f_pred if f_pred else float("nan")

            print(f"  {dname:<18} 정답 {tot_gt:>6,} | "
                  f"발견율 {det_rec:.3f} 정밀도 {det_prec:.3f}", end="")
            if has_pose and nc > 1 and f_gt:
                print(f" | 쓰러짐 {f_gt:>4,}개 재현율 {f_rec:.3f} 정밀도 {f_prec:.3f}")
            else:
                print()
            rows.append(dict(model=tag, nc=nc, domain=dname, gt=tot_gt,
                             det_recall=round(det_rec, 4),
                             det_precision=round(det_prec, 4),
                             fallen_gt=f_gt,
                             fallen_recall=round(f_rec, 4) if f_gt else "",
                             fallen_precision=round(f_prec, 4) if f_gt else ""))
        print()

    if rows:
        OUT_CSV.parent.mkdir(exist_ok=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"저장: {OUT_CSV}")

        # 합격 기준 판정 — 마지막 모델 기준
        last = rows[-1]["model"]
        crit = [("NOMAD 탐지", "det_recall", 0.60),
                ("WiSARD 탐지", "det_recall", 0.85),
                ("SARD 자세(test)", "fallen_recall", 0.90),
                ("NOMAD 자세(val)", "fallen_recall", 0.50)]
        print(f"\n=== 합격 기준 판정 — {last} ===")
        for dname, key, target in crit:
            hit = [r for r in rows if r["model"] == last and r["domain"] == dname]
            if not hit or hit[0][key] == "":
                print(f"  {dname:<18} {key:<15} 측정 불가")
                continue
            v = hit[0][key]
            mark = "통과" if v >= target else "미달"
            print(f"  {dname:<18} {key:<15} {v:.3f} / {target:.2f}  {mark}")


if __name__ == "__main__":
    main()
