"""
autonomous_patrol.py — 8층 지도 + NBV 플래너 기반 자율 정찰 루프 (STEP 12)
전제:
  - 언리얼 Play 상태
  - WSL에서 SITL + airsim-copter 실행 중
실행: python autonomous_patrol.py
차이점(patrol_detect.py 대비):
  - 고정된 사각형 경로 대신, 매 스텝 8층 지도를 보고 NBV 플래너가 다음 목표를 스스로 결정
"""
import time

import cosysairsim as airsim
from pymavlink import mavutil

from map_manager import SituationMap
from planner import NextBestViewPlanner
from patrol_detect import (
    connect_mavlink, send_waypoint, arm_and_takeoff, wait_arrival,
    DetectionThread, MAVLINK_UDP, YOLO_MODEL, MAP_DIR,
)
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────────────
CRUISE_ALT   = 8.0     # 순항 고도 (m)
NUM_STEPS    = 12      # 최대 관측 스텝 수
ARRIVAL_TOL  = 2.0      # 목표 도착 판정 거리 (m)
# ──────────────────────────────────────────────────────────────────────


def run_autonomous(drone, ac_client=None):
    """이미 연결·Arm·GUIDED·이륙까지 완료된 drone(MAVLink 연결)을 받아
    8층 지도 + NBV 플래너 기반 자율 정찰 루프를 실행하고 RTL까지 수행한다.
    run_experiment.py 등 오케스트레이터가 자체 MAVLink 연결을 재사용할 때 호출."""
    if ac_client is None:
        print("[AirSim] 연결 중...")
        ac_client = airsim.MultirotorClient()
        ac_client.confirmConnection()
        print("[AirSim] 연결됨")

    print(f"[YOLO] 모델 로드: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    sit_map = SituationMap(width=40, height=40, cell_size=1.0, origin_offset=(10.0, 10.0))
    planner = NextBestViewPlanner(sit_map, cruise_alt=CRUISE_ALT)

    det = DetectionThread(model, sit_map)
    det.start()
    print("[탐지] 스레드 시작")

    print(f"\n[NBV] 자율 정찰 시작 (최대 {NUM_STEPS} 스텝)")
    x, y = 0.0, 0.0
    sit_map.update_from_detection([], x, y, CRUISE_ALT)  # 시작 위치 관측 처리

    for step in range(1, NUM_STEPS + 1):
        result = planner.next_waypoint(x, y)
        if result is None:
            print(f"[NBV] step {step}: 탐색할 후보 없음 → 정찰 종료")
            break

        tx, ty, tz, ctype = result
        print(f"\n[NBV] step {step}/{NUM_STEPS}: 유형={ctype} | "
              f"현재=({x:.1f},{y:.1f}) → 목표=({tx:.1f},{ty:.1f})")

        send_waypoint(drone, tx, ty, tz)
        wait_arrival(drone, tx, ty, tz, threshold=ARRIVAL_TOL)
        x, y = tx, ty

        print(f"[NBV] step {step} 도착 ✓ | 탐지 FPS: {det.fps:.1f} | "
              f"미관측 {sit_map.get_unobserved_ratio()*100:.0f}%")
        sit_map.save(f"{MAP_DIR}/nbv_step_{step:02d}.png")

    print("\n[비행] 정찰 완료. 복귀(RTL)...")
    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(10)

    det.stop()
    sit_map.save(f"{MAP_DIR}/nbv_final.png")
    print(f"\n[완료] 총 {det.frame}프레임 처리 | 최종 미관측 비율: "
          f"{sit_map.get_unobserved_ratio()*100:.0f}% | 지도: {MAP_DIR}/nbv_final.png")
    print("STEP 10(8층 지도) + STEP 11(NBV 플래너) + STEP 12(통합 루프) 달성!")
    return {"frames": det.frame, "unobserved_ratio": sit_map.get_unobserved_ratio()}


# ── 독립 실행 진입점 ──────────────────────────────────────────────────
def main():
    """단독 실행(python autonomous_patrol.py)용: 자체적으로 연결·Arm·이륙까지 수행."""
    print("[AirSim] 연결 중...")
    ac = airsim.MultirotorClient()
    ac.confirmConnection()
    print("[AirSim] 연결됨")

    drone = connect_mavlink()

    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 1, 4, 0, 0, 0, 0, 0   # GUIDED
    )
    time.sleep(1)

    arm_and_takeoff(drone, altitude=CRUISE_ALT)

    run_autonomous(drone, ac)


if __name__ == "__main__":
    main()
