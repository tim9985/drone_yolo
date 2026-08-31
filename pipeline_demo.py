"""
pipeline_demo.py — 탐지·추적·Depth·좌표변환 통합 검증 (작업 3·4 검증용)

드론(ComputerVision 카메라)이 직선으로 이동하며 정지한 마네킹들을 스쳐 지나가는
상황을 만들어 두 모드를 같은 프레임 열에서 비교한다.

  모드 A: 매 프레임 YOLO (기존 구조)
  모드 B: YOLO 저주기(기본 2Hz 상당) + 추적 고주기 (작업 3 구조)

측정: 실효 FPS, YOLO 호출 횟수, track_id 유지율
부가: 모드 B 프레임 일부를 debug_overlay 로 저장 (작업 4 예시)

전제: 언리얼이 settings_coord.json (ComputerVision, ImageType 0/1/5) 으로 실행 중.
실행: python pipeline_demo.py [--frames 40] [--detect-every 5]
"""
import argparse
import math
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
from cosysairsim.utils import euler_to_quaternion
from ultralytics import YOLO

from coord_transform import pixel_to_ned
from tracking import MultiObjectTracker, boxes_from_yolo
from debug_overlay import save_debug_frame, format_coord_line
from map_manager import SituationMap

BASE_DIR = Path(__file__).resolve().parent
IMG_W, IMG_H, FOV = 1920, 1080, 54.0
CAM_NADIR = (0.0, -math.pi / 2, 0.0)
ALT = 20.0
YOLO_IMGSZ = 960
YOLO_WEIGHTS = "yolov8s_stage1_all.pt"   # NOMAD(배우1~30) + WiSARD(9월·1월) 학습
YOLO_CONF = 0.15                         # 근거: threshold_tuning.csv (patrol_detect.py 주석 참조)


def capture(client):
    reqs = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)]
    r = client.simGetImages(reqs)
    img = np.frombuffer(r[0].image_data_uint8, dtype=np.uint8).reshape(r[0].height, r[0].width, 3)
    depth = np.array(r[1].image_data_float, dtype=np.float32).reshape(r[1].height, r[1].width)
    return img, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--detect-every", type=int, default=5,
                    help="모드 B에서 N프레임마다 YOLO 호출 (5 ≈ 10Hz 루프에서 2Hz)")
    ap.add_argument("--overlay-every", type=int, default=10)
    args = ap.parse_args()

    client = airsim.VehicleClient()
    client.confirmConnection()
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                                              euler_to_quaternion(*CAM_NADIR)))

    # 마네킹들이 있는 구간을 가로지르는 직선 경로
    poses = client.simListInstanceSegmentationPoses(ned=True)
    objs = client.simListInstanceSegmentationObjects()
    people = [(objs[i], poses[i].position) for i, n in enumerate(objs)
              if n.startswith("SkeletalMeshActor")]
    if people:
        xs = [p.x_val for _, p in people]
        ys = [p.y_val for _, p in people]
        x0, x1 = min(xs) - 3, max(xs) + 3
        ymid = float(np.median(ys))
    else:
        x0, x1, ymid = -5.0, 20.0, 0.0
    ground_z = max((p.z_val for _, p in people), default=0.0)
    print(f"경로: x {x0:.1f} → {x1:.1f}, y={ymid:.1f}, 고도 {ALT}m, 사람 {len(people)}명")

    model = YOLO(str(BASE_DIR / "weights" / YOLO_WEIGHTS))
    path = [(x0 + (x1 - x0) * i / max(args.frames - 1, 1), ymid) for i in range(args.frames)]

    def fly_capture(i):
        x, y = path[i]
        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(x, y, ground_z - ALT), euler_to_quaternion(0, 0, 0)), True)
        time.sleep(0.05)
        return (x, y, ground_z - ALT), capture(client)

    # 워밍업 (모델 로드 + CUDA 초기화 비용 제외)
    _, (img0, _) = fly_capture(0)
    for _ in range(3):
        model(img0, verbose=False, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)

    # ── 모드 A: 매 프레임 YOLO ──
    t0 = time.perf_counter()
    n_yolo_a, n_det_a = 0, 0
    for i in range(args.frames):
        _, (img, _) = fly_capture(i)
        res = model(img, verbose=False, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
        n_yolo_a += 1
        n_det_a += len(res[0].boxes)
    ta = time.perf_counter() - t0

    # ── 모드 B: 저주기 탐지 + 추적 ──
    mot = MultiObjectTracker()
    sit_map = SituationMap(width=60, height=60, cell_size=1.0, origin_offset=(20.0, 20.0))
    id_hist, n_yolo_b = [], 0
    saved = []
    t0 = time.perf_counter()
    for i in range(args.frames):
        drone_ned, (img, depth) = fly_capture(i)
        if i % args.detect_every == 0:
            res = model(img, verbose=False, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
            dets = boxes_from_yolo(res[0], model.names)
            tracks = mot.update_with_detections(img, dets, i)
            n_yolo_b += 1
            do_det = True
        else:
            dets, do_det = [], False
            tracks = mot.update_tracking_only(img, i)

        # 추적 객체 → 실좌표 → 8층 지도
        entries, est = [], None
        for t in tracks:
            cx, cy = t.center
            d = float(depth[min(max(int(cy), 0), depth.shape[0]-1),
                            min(max(int(cx), 0), depth.shape[1]-1)])
            e = {"class": t.cls_name, "confidence": t.confidence}
            if np.isfinite(d) and 0 < d < 500:
                pt = pixel_to_ned(cx, cy, IMG_W, IMG_H, FOV, drone_ned, (0, 0, 0), d,
                                  camera_orientation=CAM_NADIR, depth_type="planar")
                e["rel_x"] = pt.x - drone_ned[0]
                e["rel_y"] = pt.y - drone_ned[1]
                e["height_m"] = max(0.0, ground_z - pt.z)   # HEIGHT 층 입력
                est = est or (pt.x, pt.y)
            entries.append(e)
        sit_map.step(dt=1.0)
        sit_map.update_from_detection(entries, drone_ned[0], drone_ned[1], ALT)
        id_hist.append({t.track_id for t in tracks})

        if args.overlay_every and i % args.overlay_every == 0:
            lines = []
            if est:
                lines.append(format_coord_line(est_ned=est))
            lines.append(f"mode B | YOLO {n_yolo_b}/{i+1} frames | backend {mot.backend}")
            p = save_debug_frame(img, i,
                                 detections=[{"class": d["class"], "confidence": d["confidence"],
                                              "bbox": d["bbox"]} for d in dets],
                                 tracks=tracks, depth=depth, info_lines=lines)
            saved.append(p)
    tb = time.perf_counter() - t0

    # ── 결과 ──
    all_ids = set().union(*id_hist) if id_hist else set()
    life = {i: sum(1 for s in id_hist if i in s) for i in all_ids}
    long_lived = [i for i, n in life.items() if n >= args.detect_every]
    print(f"\n=== 모드 비교 ({args.frames} 프레임) ===")
    print(f"{'':22} {'소요(s)':>9} {'FPS':>7} {'YOLO 호출':>9}")
    print(f"{'A 매프레임 탐지':22} {ta:>9.2f} {args.frames/ta:>7.2f} {n_yolo_a:>9}")
    print(f"{'B 저주기탐지+추적':22} {tb:>9.2f} {args.frames/tb:>7.2f} {n_yolo_b:>9}")
    print(f"\nFPS 개선: {(args.frames/tb)/(args.frames/ta):.2f}배 | "
          f"YOLO 호출 {100*(1-n_yolo_b/max(n_yolo_a,1)):.0f}% 감소")
    print(f"\n=== 추적 안정성 ===")
    print(f"생성된 track_id {len(all_ids)}개 | {args.detect_every}프레임 이상 유지 {len(long_lived)}개")
    if life:
        top = sorted(life.items(), key=lambda kv: -kv[1])[:5]
        print("  최장 유지 track_id:", ", ".join(f"ID{i}={n}프레임" for i, n in top))
        print(f"  평균 유지 길이 {np.mean(list(life.values())):.1f}프레임 "
              f"(전체 {args.frames}프레임 중)")
    print(f"\n지도 미관측 비율: {sit_map.get_unobserved_ratio()*100:.0f}%")
    sit_map.save(str(BASE_DIR / "maps" / "pipeline_demo_map.png"))
    print(f"오버레이 {len(saved)}장 저장: results/debug_frame_*.png")


if __name__ == "__main__":
    main()
