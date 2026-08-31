"""
gsd_calc.py — 지상 분해능(GSD) + 탐색 효율 기반 운용 조건 계산표

  지상 촬영 폭 W = 2 × 고도 × tan(FOV/2)
  픽셀당 실거리 GSD = W / 영상 가로 픽셀수
  대상 픽셀 수 = 대상 실제 크기 / GSD

탐색 효율 (100m×100m 구역, 비행속도 3m/s, 중첩 20% 가정):
  lane_width_m    = ground_width_m × 0.8
  passes          = ceil(100 / lane_width_m)
  search_time_min = (passes × 100) / 3.0 / 60
  battery_ok      = search_time_min <= 10
  altitude_safe   = altitude_m >= 15   (장애물 안전 하한)

실행: python gsd_calc.py [--e2e "640=6.4,960=6.1,1280=5.2"]
출력: gsd_table.csv (전체 조합)
      operating_conditions.csv (사람>=20px AND e2e>=5FPS AND battery_ok AND altitude_safe)
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
OUT_CSV = BASE_DIR / "metrics" / "gsd_table.csv"
OUT_OPS_CSV = BASE_DIR / "metrics" / "operating_conditions.csv"

PERSON_W = 0.5    # 사람 위에서 본 폭 (m)
VEHICLE_W = 4.5   # 차량 길이 (m)
PIXEL_THRESHOLD = 20  # YOLO 안정 탐지 최소 픽셀

ALTITUDES = range(10, 51, 5)        # 10~50m, 5m 간격
FOVS = (60, 90, 120)                # 도
WIDTHS = (640, 960, 1280)           # 영상 가로 픽셀 (= 추론 imgsz와 대응)

AREA_SIDE_M = 100.0   # 탐색 구역 한 변 (m)
OVERLAP = 0.8         # 중첩 20% → 유효 폭 80%
SPEED_MS = 3.0        # 비행 속도 (m/s)
BATTERY_LIMIT_MIN = 10.0
ALT_SAFE_MIN = 15     # 장애물 안전 고도 하한 (m)


def build_table(e2e_by_width):
    rows = []
    for alt in ALTITUDES:
        for fov in FOVS:
            ground_w = 2 * alt * math.tan(math.radians(fov / 2))
            lane_w = ground_w * OVERLAP
            passes = math.ceil(AREA_SIDE_M / lane_w)
            search_min = (passes * AREA_SIDE_M) / SPEED_MS / 60.0
            for px_w in WIDTHS:
                gsd = ground_w / px_w                  # m/pixel
                person_px = PERSON_W / gsd
                vehicle_px = VEHICLE_W / gsd
                e2e = e2e_by_width.get(px_w)
                rows.append({
                    "altitude_m": alt,
                    "fov_deg": fov,
                    "image_width_px": px_w,
                    "ground_width_m": round(ground_w, 1),
                    "gsd_cm_per_px": round(gsd * 100, 2),
                    "person_px": round(person_px, 1),
                    "vehicle_px": round(vehicle_px, 1),
                    "person_detectable": person_px >= PIXEL_THRESHOLD,
                    "lane_width_m": round(lane_w, 1),
                    "passes": passes,
                    "search_time_min": round(search_min, 1),
                    "battery_ok": search_min <= BATTERY_LIMIT_MIN,
                    "altitude_safe": alt >= ALT_SAFE_MIN,
                    "e2e_fps_measured": e2e if e2e is not None else "",
                    "e2e_ok": (e2e >= 5.0) if e2e is not None else "",
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", default="",
                        help='실측 e2e FPS 매핑. 예: "640=6.4,960=6.1,1280=5.2" (yolov8s_visdrone 기준)')
    args = parser.parse_args()

    e2e_by_width = {}
    if args.e2e:
        for part in args.e2e.split(","):
            k, v = part.split("=")
            e2e_by_width[int(k)] = float(v)

    rows = build_table(e2e_by_width)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4중 제약 필터 (e2e 실측값이 주어진 경우에만 운용조건 CSV 생성)
    ops = [r for r in rows
           if r["person_detectable"] and r["battery_ok"] and r["altitude_safe"]
           and r["e2e_ok"] is True]
    if e2e_by_width:
        with open(OUT_OPS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(ops)

    print(f"전체 {len(rows)}개 조합 | 사람 {PIXEL_THRESHOLD}px 이상: "
          f"{sum(1 for r in rows if r['person_detectable'])}개 | "
          f"4중 제약 통과: {len(ops)}개\n")
    hdr = (f"{'고도':>4} {'FOV':>4} {'해상도':>6} {'촬영폭m':>7} {'사람px':>7} "
           f"{'레인폭m':>7} {'왕복':>4} {'탐색분':>6} {'e2e':>5}")
    print("=== 4중 제약(사람px·e2e·배터리·고도안전) 통과 조합 ===")
    print(hdr)
    for r in ops:
        print(f"{r['altitude_m']:>4} {r['fov_deg']:>4} {r['image_width_px']:>6} "
              f"{r['ground_width_m']:>7.1f} {r['person_px']:>7.1f} "
              f"{r['lane_width_m']:>7.1f} {r['passes']:>4} {r['search_time_min']:>6.1f} "
              f"{r['e2e_fps_measured']:>5}")
    print(f"\n저장: {OUT_CSV}" + (f", {OUT_OPS_CSV}" if e2e_by_width else ""))
    return rows


if __name__ == "__main__":
    main()
