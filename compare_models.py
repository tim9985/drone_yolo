"""
compare_models.py — VisDrone 실측 샘플에서 두 모델의 실제 탐지 성능 비교 (작업 4)

yolov8n(COCO)과 yolov8s(VisDrone 파인튜닝)를 같은 항공뷰 이미지에 적용해
"모델 교체가 실제로 효과가 있었다"를 증명한다.

실행: python compare_models.py
출력:
  - comparison/visdrone_XX_<이름>_side_by_side.png  (좌: COCO, 우: VisDrone 모델)
  - model_comparison.csv                            (이미지별 탐지 개수·평균 신뢰도)
  - 콘솔 요약 표
"""
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
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "data" / "raw" / "visdrone_samples"
COMPARISON_DIR = BASE_DIR / "comparison"
OUT_CSV = BASE_DIR / "metrics" / "model_comparison.csv"
IMGSZ = 960

# VisDrone 모델 클래스 중 사람/차량 계열 (COCO와 명칭이 달라 그룹으로 비교)
PERSON_LIKE = {"pedestrian", "people", "person"}
VEHICLE_LIKE = {"car", "van", "truck", "bus", "motor", "motorcycle", "bicycle",
                "tricycle", "awning-tricycle"}


def summarize(results, names):
    dets = [(names[int(b.cls[0])], float(b.conf[0])) for b in results.boxes]
    n_person = sum(1 for c, _ in dets if c in PERSON_LIKE)
    n_vehicle = sum(1 for c, _ in dets if c in VEHICLE_LIKE)
    mean_conf = float(np.mean([cf for _, cf in dets])) if dets else 0.0
    return len(dets), n_person, n_vehicle, mean_conf


def main():
    models = {
        "yolov8n_coco": YOLO(str(BASE_DIR / "yolov8n.pt")),
        "yolov8s_visdrone": YOLO(str(BASE_DIR / "weights" / "yolov8s_visdrone.pt")),
    }
    COMPARISON_DIR.mkdir(exist_ok=True)

    images = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"샘플 이미지 없음: {SAMPLES_DIR}")

    rows = []
    print(f"{'이미지':32} {'모델':18} {'전체':>4} {'사람':>4} {'차량':>4} {'평균신뢰도':>8}")
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        panels = []
        for mname, model in models.items():
            res = model(img, verbose=False, imgsz=IMGSZ)[0]
            total, n_p, n_v, mconf = summarize(res, model.names)
            rows.append({"image": img_path.name, "model": mname, "total": total,
                         "person": n_p, "vehicle": n_v, "mean_conf": round(mconf, 3)})
            print(f"{img_path.name:32} {mname:18} {total:>4} {n_p:>4} {n_v:>4} {mconf:>8.3f}")

            panel = res.plot()
            label = f"{mname}: {total} det (person {n_p} / vehicle {n_v})"
            cv2.putText(panel, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)

        side = np.hstack(panels)
        out = COMPARISON_DIR / f"visdrone_{idx:02d}_{img_path.stem}_side_by_side.png"
        cv2.imwrite(str(out), side)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "model", "total", "person", "vehicle", "mean_conf"])
        w.writeheader()
        w.writerows(rows)

    # 모델별 합계
    print("\n=== 합계 ===")
    for mname in models:
        sub = [r for r in rows if r["model"] == mname]
        tp = sum(r["person"] for r in sub)
        tv = sum(r["vehicle"] for r in sub)
        tt = sum(r["total"] for r in sub)
        confs = [r["mean_conf"] for r in sub if r["total"] > 0]
        print(f"{mname:18} 전체 {tt:4d} | 사람 {tp:4d} | 차량 {tv:4d} | "
              f"평균신뢰도 {np.mean(confs) if confs else 0:.3f}")
    print(f"\n저장: {OUT_CSV}, {COMPARISON_DIR}/visdrone_XX_*_side_by_side.png")


if __name__ == "__main__":
    main()
