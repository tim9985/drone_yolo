"""
collect_data.py — 드론 시점 RGB / Depth / Segmentation + 상태 수집 (M5)
전제: 언리얼 Play 상태 (비행 중이 아니어도 됨)
실행: python collect_data.py            → 1회 캡처
      python collect_data.py --loop 10  → 2초 간격 10회 캡처 (비행 중 데이터셋 수집용)
출력: dataset/ 폴더에 rgb_XXX.png, depth_XXX.npy, seg_XXX.png, state_XXX.json
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import euler_to_quaternion

OUT_DIR = "data/misc/depth_samples"


def capture(client: airsim.MultirotorClient, idx: int) -> None:
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    ])

    # RGB
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    rgb = rgb.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite(f"{OUT_DIR}/rgb_{idx:04d}.png", rgb)

    # Depth (미터 단위 float 2D 배열 — ground truth)
    depth = airsim.list_to_2d_float_array(
        responses[1].image_data_float, responses[1].width, responses[1].height)
    np.save(f"{OUT_DIR}/depth_{idx:04d}.npy", depth)

    # Segmentation (ground truth 라벨 영상)
    seg = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
    seg = seg.reshape(responses[2].height, responses[2].width, 3)
    cv2.imwrite(f"{OUT_DIR}/seg_{idx:04d}.png", seg)

    # 드론 상태 (위치·자세·GPS)
    st = client.getMultirotorState()
    pos = st.kinematics_estimated.position
    ori = st.kinematics_estimated.orientation
    gps = client.getGpsData().gnss.geo_point
    state = {
        "position_ned": {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
        "orientation_quat": {"w": ori.w_val, "x": ori.x_val, "y": ori.y_val, "z": ori.z_val},
        "gps": {"lat": gps.latitude, "lon": gps.longitude, "alt": gps.altitude},
        "timestamp": time.time(),
    }
    with open(f"{OUT_DIR}/state_{idx:04d}.json", "w") as f:
        json.dump(state, f, indent=2)

    print(f"[{idx:04d}] 저장 완료 | 위치 x={pos.x_val:.1f} y={pos.y_val:.1f} z={pos.z_val:.1f} "
          f"| depth 범위 {depth.min():.1f}~{min(depth.max(), 999):.1f}m")


def run_collect(loop: int = 1, interval: float = 2.0) -> None:
    """다른 스크립트(run_experiment.py 등)에서 argparse 없이 직접 호출하기 위한 함수."""
    os.makedirs(OUT_DIR, exist_ok=True)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print(f"수집 시작 → {OUT_DIR}/ (총 {loop}회)")

    for i in range(loop):
        capture(client, i)
        if i < loop - 1:
            time.sleep(interval)

    print("수집 완료. rgb_0000.png 를 열어 드론 시점이 보이면 M5 통과")


# ═══════════════════════════════════════════════════════════════════════
# 합성 데이터 수집 (--synth) — ComputerVision 모드 전용
# ═══════════════════════════════════════════════════════════════════════
SYNTH_DIR = "data/det/synth/raw"
SYNTH_ALTITUDES = (15.0, 20.0, 25.0, 30.0)   # m (지면 기준 상대 고도)
SYNTH_CAM_PITCHES = (-90.0, -45.0)           # 수직 하방 / 45도
NEGATIVE_RATIO = 0.15                        # 사람이 없는 배경 프레임 비율


def _quat(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    import math
    # cosysairsim은 to_quaternion 대신 euler_to_quaternion(roll, pitch, yaw) [rad] 제공
    return euler_to_quaternion(math.radians(roll_deg), math.radians(pitch_deg),
                               math.radians(yaw_deg))


def _teleport(client, x, y, z_ned, yaw_deg=0.0):
    """ComputerVision 모드에서 카메라(가상 기체)를 NED 좌표로 이동. z_ned는 NED z(아래+)."""
    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(x, y, z_ned), _quat(yaw_deg=yaw_deg)), True)


def _set_cam_pitch(client, pitch_deg):
    client.simSetCameraPose(
        "0", airsim.Pose(airsim.Vector3r(0, 0, 0), _quat(pitch_deg=pitch_deg)))


def _capture_pair(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    ])
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    rgb = rgb.reshape(responses[0].height, responses[0].width, 3)
    seg = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
    seg = seg.reshape(responses[1].height, responses[1].width, 3)
    return rgb, seg


def run_synth_collect(count=200, out_dir=SYNTH_DIR, seed=7, log=print,
                      altitudes=None, pitches=None):
    """다양한 고도/각도/위치에서 RGB+seg 쌍 자동 수집 (ComputerVision 모드 전제).

    사람 액터의 실제 NED 좌표를 런타임에 조회해 그 상공을 촬영하므로,
    언리얼 월드 원점과 AirSim 원점의 오프셋에 영향받지 않는다.
    (place_persons.py의 UE 좌표를 그대로 쓰면 프레임에 사람이 안 들어옴)
    """
    import random
    random.seed(seed)

    alt_list = tuple(altitudes) if altitudes else SYNTH_ALTITUDES
    pitch_list = tuple(pitches) if pitches else SYNTH_CAM_PITCHES
    log(f"[synth] 고도 {alt_list} / 카메라각 {pitch_list}")

    from set_segmentation_ids import build_color_map

    os.makedirs(out_dir, exist_ok=True)
    client = airsim.VehicleClient()
    client.confirmConnection()

    # 인스턴스 segmentation 색상 매핑 + 사람 위치 조회
    cmap = build_color_map(client, log=log)
    persons = cmap["person"]
    ground_z = max(e["ned"][2] for e in persons)  # NED z (아래 +) → 지면 근처 값
    log(f"[synth] 사람 {len(persons)}명, 지면 z(NED)≈{ground_z:.2f}m")

    # 시간대/조명 변화 시도 (Blocks 맵은 미지원일 수 있음)
    try:
        client.simSetTimeOfDay(True, start_datetime="2026-07-30 14:00:00",
                               celestial_clock_speed=0)
        log("[synth] TimeOfDay 설정 시도 완료")
    except Exception as e:
        log(f"[synth] TimeOfDay 미지원(무시): {e}")

    t0 = time.time()
    n_person_frames = 0
    for i in range(count):
        alt = random.choice(alt_list)
        pitch = random.choice(pitch_list)
        yaw = random.uniform(0, 360)
        z_ned = ground_z - alt  # 지면 위 alt 미터

        if random.random() < NEGATIVE_RATIO:
            # 배경 전용 프레임 (사람 없음) — 오탐 억제 학습용
            tx = random.uniform(-25.0, 25.0)
            ty = random.uniform(-25.0, 25.0)
            target = None
        else:
            target = random.choice(persons)
            px, py = target["ned"][0], target["ned"][1]
            # 지터 한계는 화면 세로 방향 지상 반폭으로 정해진다:
            #   가로 hFOV=60° → half_w = alt*tan(30°) = 0.577*alt
            #   세로는 720/1280 비율 → half_h = 0.325*alt  (이쪽이 더 좁음)
            # yaw가 무작위라 어느 축으로든 밀릴 수 있으므로 좁은 쪽(0.325)의 절반 이내로 제한.
            jitter = 0.16 * alt
            if pitch == -45.0:
                # 45도 전방 하방: 지상 주시점이 고도와 같은 수평거리만큼 정면(+x)에 놓임
                yaw = 0.0
                tx = px - alt + random.uniform(-jitter, jitter)
                ty = py + random.uniform(-jitter, jitter)
            else:
                tx = px + random.uniform(-jitter, jitter)
                ty = py + random.uniform(-jitter, jitter)
            n_person_frames += 1

        _set_cam_pitch(client, pitch)
        _teleport(client, tx, ty, z_ned, yaw_deg=yaw)
        # 텔레포트 직후 첫 캡처는 이전 위치의 프레임이 나올 수 있다(렌더 1프레임 지연).
        # 대기 후 한 번 버리고 다시 캡처해 조준 위치와 영상을 일치시킨다.
        time.sleep(0.25)
        _capture_pair(client)
        rgb, seg = _capture_pair(client)

        cv2.imwrite(f"{out_dir}/rgb_{i:04d}.png", rgb)
        cv2.imwrite(f"{out_dir}/seg_{i:04d}.png", seg)
        with open(f"{out_dir}/meta_{i:04d}.json", "w", encoding="utf-8") as f:
            json.dump({"ned": [tx, ty, z_ned], "altitude_m": alt,
                       "cam_pitch_deg": pitch, "cam_yaw_deg": yaw,
                       "target": target["name"] if target else None,
                       "timestamp": time.time()}, f)

        if (i + 1) % 20 == 0:
            rate = (i + 1) / (time.time() - t0)
            log(f"[synth] {i+1}/{count} 수집 ({rate:.1f}장/s)")

    elapsed = time.time() - t0
    log(f"[synth] 수집 완료: {count}장 쌍 → {out_dir} "
        f"(사람 조준 {n_person_frames} / 배경 {count - n_person_frames}) "
        f"{elapsed:.0f}s, {count/elapsed:.1f}장/s")

    # 수집 조건 메타데이터 — 학습 시 추론 조건과의 정합성을 추적하기 위해 남긴다
    fov = 60.0
    info = {
        "count": count,
        "altitudes_m": list(alt_list),
        "camera_pitches_deg": list(pitch_list),
        "fov_degrees": fov,
        "image_size": [1280, 720],
        "gsd_cm_per_px_by_alt": {
            str(a): round((2 * a * np.tan(np.radians(fov / 2))) / 1280 * 100, 4)
            for a in alt_list
        },
        "person_instances": len(persons),
        "person_targeted_frames": n_person_frames,
        "background_frames": count - n_person_frames,
        "negative_ratio": NEGATIVE_RATIO,
        "ground_z_ned": ground_z,
        "elapsed_s": round(elapsed, 1),
        "note": "자세 분포는 place_persons.py 배치 규약(서있음→쓰러짐→가림→보정표본 순)에 따름. "
                "라벨 통계는 dataset_stats.py 참조.",
    }
    info_path = os.path.join(os.path.dirname(out_dir), "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log(f"[synth] 수집 조건 기록: {info_path}")

    return {"count": count, "elapsed_s": elapsed, "person_frames": n_person_frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1, help="캡처 횟수 (기본 1회)")
    parser.add_argument("--interval", type=float, default=2.0, help="캡처 간격(초)")
    parser.add_argument("--synth", type=int, metavar="N",
                        help="합성 데이터 N장 수집 (ComputerVision 모드 전제)")
    args = parser.parse_args()
    if args.synth:
        run_synth_collect(count=args.synth)
    else:
        run_collect(loop=args.loop, interval=args.interval)


if __name__ == "__main__":
    main()
