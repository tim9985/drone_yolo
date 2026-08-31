"""
coord_validate.py — 픽셀→실좌표 변환 정확도 측정 (작업 2 검증)

AirSim ground truth 와 비교해 오차를 미터 단위로 측정한다.
  1. 대상 오브젝트의 실제 NED 좌표를 simListInstanceSegmentationPoses 로 확보
  2. 드론(ComputerVision 카메라)을 여러 위치·고도·자세로 이동
  3. 대상의 화면 픽셀 위치는 **segmentation 마스크 중심**으로 정확히 특정
     (YOLO 탐지가 안 되는 큐브/실린더도 검증 가능)
  4. 그 픽셀의 depth 로 pixel_to_ned 실행 → ground truth 와 XY 오차 계산
  5. 자세보정 ON/OFF 를 같은 프레임에서 각각 계산해 직접 비교

전제: 언리얼이 settings_coord.json (ComputerVision, ImageType 0/1/5 @1280x720 FOV90)
      으로 실행 중이어야 한다.

실행: python coord_validate.py [--samples 16]
출력: coord_accuracy.csv, 콘솔 요약
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion, load_colormap

from coord_transform import pixel_to_ned, ned_to_gps

BASE_DIR = Path(__file__).resolve().parent
OUT_CSV = BASE_DIR / "metrics" / "coord_accuracy.csv"

IMG_W, IMG_H, FOV = 1920, 1080, 54.0
CAM_NADIR = (0.0, -math.pi / 2, 0.0)
ORIGIN_GPS = (35.1796, 129.0756)

# 대상 후보: 배치한 잔해 큐브와 마네킹. 큐브는 상면 중심이 마스크 중심과 잘 일치해
# ground truth 비교가 깔끔하다.
TARGET_PATTERNS = ("Cube", "Cylinder", "Cone", "SkeletalMeshActor")


def capture(client):
    reqs = [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)]
    client.simGetImages(reqs)          # 텔레포트 직후 첫 프레임은 버린다
    r = client.simGetImages(reqs)
    seg = np.frombuffer(r[0].image_data_uint8, dtype=np.uint8).reshape(r[0].height, r[0].width, 3)
    depth = np.array(r[1].image_data_float, dtype=np.float32).reshape(r[1].height, r[1].width)
    return seg, depth


def mask_centroid(seg, rgb):
    m = np.all(seg == np.array(rgb, dtype=np.uint8), axis=2)
    n = int(m.sum())
    if n < 60:
        return None, n
    ys, xs = np.where(m)
    return (float(xs.mean()), float(ys.mean())), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=16, help="측정 표본 수")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed)

    client = airsim.VehicleClient()
    client.confirmConnection()

    objs = client.simListInstanceSegmentationObjects()
    poses = client.simListInstanceSegmentationPoses(ned=True)
    cmap = load_colormap()

    # 원점 부근(±40m)에 있는 후보만 사용
    cands = []
    for i, name in enumerate(objs):
        if not any(k.lower() in name.lower() for k in TARGET_PATTERNS):
            continue
        p = poses[i].position
        if not (math.isfinite(p.x_val) and math.isfinite(p.y_val)):
            continue
        if abs(p.x_val) > 40 or abs(p.y_val) > 40:
            continue
        cands.append({"name": name, "idx": i,
                      "rgb": [int(v) for v in cmap[i]],
                      "ned": (float(p.x_val), float(p.y_val), float(p.z_val))})
    if not cands:
        raise SystemExit("원점 부근에서 대상 오브젝트를 찾지 못함")
    print(f"대상 후보 {len(cands)}개 (원점 ±40m)")
    for c in cands[:8]:
        print(f"  {c['name']:34} ned=({c['ned'][0]:7.2f},{c['ned'][1]:7.2f},{c['ned'][2]:6.2f})")

    rows = []
    n_samples = 0          # 표본 수 (rows 는 표본당 2행: 자세보정 ON/OFF)
    attempts = 0
    while n_samples < args.samples and attempts < args.samples * 15:
        attempts += 1
        tgt = random.choice(cands)
        tx, ty, tz = tgt["ned"]

        alt = random.choice([12.0, 16.0, 20.0, 25.0, 30.0])
        # 대상이 화면에 남도록 수평 오프셋을 화각의 일부로 제한
        max_off = 0.30 * alt
        dx = random.uniform(-max_off, max_off)
        dy = random.uniform(-max_off, max_off)
        roll = math.radians(random.uniform(-15, 15))
        pitch = math.radians(random.uniform(-15, 15))
        yaw = math.radians(random.uniform(0, 360))

        drone_ned = (tx + dx, ty + dy, tz - alt)
        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(*drone_ned), euler_to_quaternion(roll, pitch, yaw)), True)
        client.simSetCameraPose(
            "0", airsim.Pose(airsim.Vector3r(0, 0, 0), euler_to_quaternion(*CAM_NADIR)))
        time.sleep(0.25)

        seg, depth = capture(client)
        c, npx = mask_centroid(seg, tgt["rgb"])
        if c is None:
            continue
        px, py = c
        d = float(depth[int(round(py)), int(round(px))])
        if not math.isfinite(d) or d <= 0 or d > 500:
            continue

        est_on = pixel_to_ned(px, py, IMG_W, IMG_H, FOV, drone_ned, (roll, pitch, yaw),
                              d, camera_orientation=CAM_NADIR, depth_type="planar",
                              apply_attitude=True)
        est_off = pixel_to_ned(px, py, IMG_W, IMG_H, FOV, drone_ned, (roll, pitch, yaw),
                               d, camera_orientation=CAM_NADIR, depth_type="planar",
                               apply_attitude=False)
        err_on = math.hypot(est_on.x - tx, est_on.y - ty)
        err_off = math.hypot(est_off.x - tx, est_off.y - ty)
        lat_on, lon_on = ned_to_gps(ORIGIN_GPS, est_on.x, est_on.y)

        n_samples += 1
        # 화면 중심에서 얼마나 떨어진 픽셀인지 — 비스듬히 본 정도의 지표
        off_center_px = math.hypot(px - IMG_W / 2, py - IMG_H / 2)
        incidence_deg = math.degrees(math.atan2(off_center_px,
                                                (IMG_W / 2) / math.tan(math.radians(FOV) / 2)))
        base = dict(sample=n_samples, target=tgt["name"],
                    drone_x=round(drone_ned[0], 3), drone_y=round(drone_ned[1], 3),
                    altitude_m=round(alt, 2),
                    roll_deg=round(math.degrees(roll), 2),
                    pitch_deg=round(math.degrees(pitch), 2),
                    yaw_deg=round(math.degrees(yaw), 2),
                    px=round(px, 1), py=round(py, 1), depth_m=round(d, 3),
                    off_center_px=round(off_center_px, 1),
                    incidence_deg=round(incidence_deg, 1),
                    mask_px=npx,
                    gt_x=round(tx, 3), gt_y=round(ty, 3))
        rows.append({**base, "attitude_corrected": True,
                     "est_x": round(est_on.x, 3), "est_y": round(est_on.y, 3),
                     "est_lat": round(lat_on, 8), "est_lon": round(lon_on, 8),
                     "error_m": round(err_on, 3)})
        rows.append({**base, "attitude_corrected": False,
                     "est_x": round(est_off.x, 3), "est_y": round(est_off.y, 3),
                     "est_lat": "", "est_lon": "",
                     "error_m": round(err_off, 3)})
        print(f"  #{base['sample']:2d} {tgt['name'][:26]:26} alt={alt:4.0f}m "
              f"r/p={math.degrees(roll):+5.1f}/{math.degrees(pitch):+5.1f}° "
              f"d={d:6.2f}m  오차 보정ON {err_on:5.2f}m / OFF {err_off:5.2f}m")

    if not rows:
        raise SystemExit("측정 표본을 얻지 못함 — 대상이 화면에 잡히지 않음")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    on = np.array([r["error_m"] for r in rows if r["attitude_corrected"]])
    off = np.array([r["error_m"] for r in rows if not r["attitude_corrected"]])
    print(f"\n=== 좌표 변환 오차 ({len(on)} 표본) ===")
    print(f"{'':14} {'평균':>8} {'중앙':>8} {'최대':>8} {'표준편차':>8}")
    print(f"{'자세보정 ON':14} {on.mean():>8.2f} {np.median(on):>8.2f} {on.max():>8.2f} {on.std():>8.2f}")
    print(f"{'자세보정 OFF':14} {off.mean():>8.2f} {np.median(off):>8.2f} {off.max():>8.2f} {off.std():>8.2f}")
    print(f"\n저장: {OUT_CSV}")

    worst = max((r for r in rows if r["attitude_corrected"]), key=lambda r: r["error_m"])
    print(f"\n최대 오차 사례: #{worst['sample']} {worst['target']}  오차 {worst['error_m']}m")
    print(f"  고도 {worst['altitude_m']}m, roll/pitch {worst['roll_deg']}/{worst['pitch_deg']}°, "
          f"픽셀 ({worst['px']},{worst['py']}), depth {worst['depth_m']}m")


if __name__ == "__main__":
    main()
