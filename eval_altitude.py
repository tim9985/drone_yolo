"""
eval_altitude.py — 촬영 고도(거리)별 탐지 성능 평가

무엇을 재는가 — **오해하기 쉬우니 먼저 읽을 것**

  NOMAD 는 같은 배우를 10m / 30m / 50m ... 여러 거리에서 찍었고, 변환된
  파일명에 그 거리가 남아 있다 (`Actor004_a10_f0001_c0.jpg` 의 `a10`).

  다만 `nomad_prep.py` 가 **사람 크기를 목표 픽셀(약 97px)로 맞춰 리샘플링**
  했기 때문에, a10 과 a30 의 사람 크기는 이미 같다 (실측 97px vs 98px).

  따라서 이 평가는 "고도가 높아 작게 찍히면 성능이 떨어지는가" 를 재는 것이
  **아니다.** 그건 크기가 정규화되면서 사라졌다. 대신 이걸 잰다:

      같은 크기로 맞춰 놓았을 때, 멀리서 찍은 사진이 더 불리한가?

  a10 은 0.4배로 **축소**돼 원본 디테일이 살아 있고, a30 은 1.2배로
  **확대**돼 없는 디테일을 늘린 것이다. 광학적 정보량이 다르다.
  이 차이가 크면 "고도를 올리면 크기를 키워도 손해" 라는 뜻이고,
  작으면 "고도를 올려 넓게 훑어도 된다" 는 근거가 된다.

  실기체 설계 고도와의 관계: 학습 목표 GSD 1.804 cm/px 는 우리 장비
  (54° · 1920px)에서 **고도 34m** 에 해당한다.

실행:
  python eval_altitude.py                                  # 운용 모델
  python eval_altitude.py --weights weights/a.pt weights/b.pt --imgsz 1280
"""
import argparse
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

BASE_DIR = Path(__file__).resolve().parent
VAL_DIRS = [BASE_DIR / "data" / "det" / "nomad_actor01_10",
            BASE_DIR / "data" / "det" / "nomad_actor11_20"]
LIST_DIR = BASE_DIR / "metrics" / "_lists"
OUT_CSV = BASE_DIR / "metrics" / "eval_altitude.csv"
DEFAULT_WEIGHTS = [BASE_DIR / "weights" / "yolov8s_stage1_all.pt"]


def build_groups():
    """파일명의 a<거리> 토큰으로 검증 이미지를 묶는다"""
    groups = defaultdict(list)
    for root in VAL_DIRS:
        vdir = root / "images" / "val"
        if not vdir.is_dir():
            continue
        for p in sorted(vdir.glob("*.jpg")):
            m = re.search(r"_a(\d+)_", p.name)
            if m:
                groups[f"a{m.group(1)}"].append(p)
    return dict(sorted(groups.items(), key=lambda kv: int(kv[0][1:])))


def write_yaml(name, images):
    """ultralytics 는 val 만 할 때도 train 키를 요구한다"""
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    txt = LIST_DIR / f"alt_{name}.txt"
    txt.write_text("\n".join(str(p) for p in images), encoding="utf-8")
    yml = LIST_DIR / f"alt_{name}.yaml"
    yml.write_text(
        f"train: {txt.as_posix()}\nval: {txt.as_posix()}\nnc: 1\nnames: ['person']\n",
        encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser(description="촬영 고도(거리)별 탐지 성능 평가")
    ap.add_argument("--weights", nargs="*", default=None, metavar="PATH")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="학습 때 쓴 값과 맞춰야 공정하다")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    from ultralytics import YOLO

    groups = build_groups()
    if not groups:
        raise SystemExit("고도 토큰(_a10_ 등)이 있는 검증 이미지를 찾지 못했습니다.")
    print("고도별 검증셋")
    for k, v in groups.items():
        n_obj = 0
        for p in v:
            lb = p.parent.parent.parent / "labels" / "val" / (p.stem + ".txt")
            if lb.exists():
                n_obj += sum(1 for _ in open(lb, encoding="utf-8", errors="ignore"))
        print(f"  {k:>5}  {len(v):>5}장 · 객체 {n_obj:>5}개")

    weights = [Path(w).resolve() for w in args.weights] if args.weights else DEFAULT_WEIGHTS
    weights = [w for w in weights if w.exists()]
    if not weights:
        raise SystemExit("평가할 가중치가 없습니다.")

    yamls = {k: write_yaml(k, v) for k, v in groups.items()}
    rows = []
    for w in weights:
        name = w.parent.parent.name if w.name in ("best.pt", "last.pt") else w.stem
        print(f"\n{'=' * 60}\n{name}  (imgsz {args.imgsz} · conf {args.conf})\n{'=' * 60}")
        model = YOLO(str(w))
        for k, yml in yamls.items():
            r = model.val(data=str(yml), imgsz=args.imgsz, conf=args.conf,
                          verbose=False, plots=False, split="val")
            b = r.box
            rows.append({"model": name, "altitude": k, "images": len(groups[k]),
                         "mAP50": round(float(b.map50), 4),
                         "mAP50_95": round(float(b.map), 4),
                         "precision": round(float(b.mp), 4),
                         "recall": round(float(b.mr), 4)})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'모델':<22} {'고도':>5} {'장수':>5} {'mAP50':>8} {'mAP50-95':>9} "
          f"{'정밀도':>8} {'재현율':>8}")
    for r in rows:
        print(f"{r['model']:<22} {r['altitude']:>5} {r['images']:>5} "
              f"{r['mAP50']:>8.3f} {r['mAP50_95']:>9.3f} "
              f"{r['precision']:>8.3f} {r['recall']:>8.3f}")

    # 같은 모델 안에서 고도 간 격차를 요약한다
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["altitude"]] = r
    print()
    for m, d in by_model.items():
        ks = sorted(d, key=lambda x: int(x[1:]))
        if len(ks) >= 2:
            lo, hi = d[ks[0]], d[ks[-1]]
            gap = hi["mAP50"] - lo["mAP50"]
            print(f"{m}: {ks[0]} → {ks[-1]} mAP50 {lo['mAP50']:.3f} → {hi['mAP50']:.3f} "
                  f"({gap:+.3f})")
    print(f"\n저장: {OUT_CSV}")
    print("주의: 사람 크기는 이미 같게 리샘플링돼 있다. 이 격차는 '크기 차이'가 아니라")
    print("      '같은 크기일 때 촬영 거리에 따른 광학 정보량 차이'다.")


if __name__ == "__main__":
    main()
