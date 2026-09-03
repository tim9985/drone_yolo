"""
eval_scale.py — 사람이 작아질수록 성능이 어떻게 떨어지는가 (= 고도 곡선)

왜 imgsz 를 흔드는가

  고도·화각·센서해상도는 독립 변수가 아니다. 셋 다 하나의 값만 바꾼다:

      GSD = 2 x 고도 x tan(화각/2) / 가로픽셀수
      사람 픽셀 = 실제 크기 / GSD

  모델이 보는 것은 결국 **입력에서 사람이 몇 픽셀이냐** 하나뿐이다.
  그리고 그 값은 추론 해상도(imgsz)로도 똑같이 바꿀 수 있다.
  imgsz 를 절반으로 낮추면 사람도 절반이 된다 — **고도를 2배로 올린 것과 같다.**

  실제 고도 상승에는 광학 정보량 손실도 따르지만, eval_altitude.py 로
  10m 대 30m 를 재보니 mAP50 0.559 대 0.579 로 **오히려 30m 가 근소하게 높았다.**
  즉 이 구간에서 광학 손실은 무시할 수준이고, 크기 효과만 보면 된다.

읽을 때 주의
  학습은 imgsz 960 으로 했다. 그보다 낮은 해상도에서 떨어지는 폭에는
  '작아져서'와 '학습 조건과 달라서'가 섞여 있다. 다만 실제로 고도를 올려도
  똑같이 학습 조건에서 멀어지므로, 운용 판단용으로는 이대로가 맞다.

실행:
  python eval_scale.py
  python eval_scale.py --weights weights/a.pt --imgszs 1280,960,640,480,320
"""
import argparse
import csv
import math
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
LIST_DIR = BASE_DIR / "metrics" / "_lists"
OUT_CSV = BASE_DIR / "metrics" / "eval_scale.csv"

# 검증셋: NOMAD(여름) + WiSARD 겨울. 사람이 있는 도메인만 쓴다.
# wisard_sept 는 사람이 0명이라 mAP 를 낼 수 없다(오탐 측정 전용).
NOMAD_VAL = [BASE_DIR / "data" / "det" / "nomad_actor01_10" / "images" / "val",
             BASE_DIR / "data" / "det" / "nomad_actor11_20" / "images" / "val"]
WISARD_VAL = BASE_DIR / "data" / "det" / "wisard" / "images" / "val"
JAN_PREFIX = "DJI_0582"          # 1월(겨울) 비행

SRC_W = 1280                     # 검증 원본 가로 픽셀
PERSON_PX_AT_SRC = 97            # 그때 사람 긴 변 (실측 중앙값)
RIG_FOV, RIG_PX = 54.0, 1920     # 예산안 실기체
PERSON_M = 1.7                   # 누운 사람 기준


def equiv_altitude(person_px_in_input):
    """모델 입력에서 사람이 이 픽셀로 보이려면, 실기체는 몇 m 에 떠야 하는가"""
    if person_px_in_input <= 0:
        return float("nan")
    gsd_cm = PERSON_M * 100 / person_px_in_input          # cm/px
    ground_w = gsd_cm * RIG_PX / 100                       # m
    return ground_w / (2 * math.tan(math.radians(RIG_FOV / 2)))


def write_yaml(name, images):
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    txt = LIST_DIR / f"scale_{name}.txt"
    txt.write_text("\n".join(str(p) for p in images), encoding="utf-8")
    yml = LIST_DIR / f"scale_{name}.yaml"
    yml.write_text(
        f"train: {txt.as_posix()}\nval: {txt.as_posix()}\nnc: 1\nnames: ['person']\n",
        encoding="utf-8")
    return yml


def main():
    ap = argparse.ArgumentParser(description="사람 크기(=고도)에 따른 성능 곡선")
    ap.add_argument("--weights", default=str(BASE_DIR / "weights" / "yolov8s_stage1_all.pt"))
    ap.add_argument("--imgszs", default="1280,960,640,480,320")
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    from ultralytics import YOLO

    nomad = []
    for d in NOMAD_VAL:
        nomad.extend(sorted(d.glob("*.jpg")))
    wis_jan = [p for p in sorted(WISARD_VAL.glob("*.jpg")) if p.name.startswith(JAN_PREFIX)]
    domains = {"nomad_여름": nomad, "wisard_겨울": wis_jan}
    domains = {k: v for k, v in domains.items() if v}
    for k, v in domains.items():
        print(f"  {k:14} {len(v):>5}장")

    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"가중치가 없습니다: {w}")
    model = YOLO(str(w))
    yamls = {k: write_yaml(k, v) for k, v in domains.items()}

    rows = []
    for imgsz in [int(x) for x in args.imgszs.split(",")]:
        person_px = PERSON_PX_AT_SRC * imgsz / SRC_W
        alt = equiv_altitude(person_px)
        for dom, yml in yamls.items():
            r = model.val(data=str(yml), imgsz=imgsz, conf=args.conf,
                          verbose=False, plots=False, split="val")
            b = r.box
            rows.append({"domain": dom, "imgsz": imgsz,
                         "person_px": round(person_px, 1),
                         "equiv_altitude_m": round(alt, 1),
                         "mAP50": round(float(b.map50), 4),
                         "recall": round(float(b.mr), 4),
                         "precision": round(float(b.mp), 4)})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'imgsz':>6} {'사람px':>7} {'환산 고도':>9} | " +
          " | ".join(f"{d:^22}" for d in domains))
    print(f"{'':>6} {'':>7} {'(54°·1920px)':>9} | " +
          " | ".join(f"{'mAP50':>7}{'재현율':>8}{'정밀도':>7}" for _ in domains))
    print("-" * (26 + 24 * len(domains)))
    for imgsz in [int(x) for x in args.imgszs.split(",")]:
        sel = [r for r in rows if r["imgsz"] == imgsz]
        head = sel[0]
        line = f"{imgsz:>6} {head['person_px']:>7.0f} {head['equiv_altitude_m']:>8.0f}m | "
        line += " | ".join(
            f"{r['mAP50']:>7.3f}{r['recall']:>8.3f}{r['precision']:>7.3f}" for r in sel)
        print(line)

    print(f"\n저장: {OUT_CSV}")
    print("환산 고도 = 실기체(54° · 1920px)에서 사람이 그 픽셀로 보이는 고도.")
    print("학습은 imgsz 960 기준이므로, 그보다 낮은 쪽의 하락에는")
    print("'작아져서'와 '학습 조건과 달라서'가 함께 들어 있다.")


if __name__ == "__main__":
    main()
