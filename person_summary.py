"""
person_summary.py — 사람 탐지 중심 발표용 요약 (작업 4)

model_comparison.csv(compare_models.py 산출물)를 읽어:
  - 사람 탐지 총합·배율 (차량/전체와 분리)
  - COCO가 사람 0명 탐지한 이미지 수
  - 이미지별 사람 탐지 비교표
  - 개선폭 top3 이미지 (발표 슬라이드용)
person_summary.csv 저장 + 콘솔 출력.

실행: python person_summary.py
"""
import csv
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
IN_CSV = BASE_DIR / "metrics" / "model_comparison.csv"
OUT_CSV = BASE_DIR / "metrics" / "person_summary.csv"

COCO = "yolov8n_coco"
VISD = "yolov8s_visdrone"


def main():
    with open(IN_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], {})[r["model"]] = int(r["person"])

    out_rows = []
    for img, d in by_image.items():
        coco_p, visd_p = d.get(COCO, 0), d.get(VISD, 0)
        out_rows.append({
            "image": img,
            "person_coco": coco_p,
            "person_visdrone": visd_p,
            "improvement": visd_p - coco_p,
            "coco_zero_person": coco_p == 0,
        })
    out_rows.sort(key=lambda r: -r["improvement"])

    total_coco = sum(r["person_coco"] for r in out_rows)
    total_visd = sum(r["person_visdrone"] for r in out_rows)
    ratio = total_visd / total_coco if total_coco else float("inf")
    n_zero = sum(1 for r in out_rows if r["coco_zero_person"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("=== 사람 탐지 요약 (imgsz 960, VisDrone 샘플 8장) ===")
    print(f"사람 탐지 총합: COCO {total_coco}명 → VisDrone {total_visd}명 ({ratio:.1f}배)")
    print(f"COCO가 사람 0명 탐지한 이미지: {n_zero}/{len(out_rows)}장 (VisDrone은 0장)")
    print()
    print(f"{'이미지':32} {'COCO':>5} {'VisDrone':>9} {'개선':>5}")
    for r in out_rows:
        mark = " ←COCO 0명" if r["coco_zero_person"] else ""
        print(f"{r['image']:32} {r['person_coco']:>5} {r['person_visdrone']:>9} "
              f"{r['improvement']:>+5}{mark}")
    print()
    print("발표 슬라이드용 개선폭 top3:")
    for i, r in enumerate(out_rows[:3], 1):
        print(f"  {i}. {r['image']} ({r['person_coco']} → {r['person_visdrone']}, "
              f"+{r['improvement']}명)")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
