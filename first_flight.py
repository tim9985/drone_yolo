"""
first_flight.py — AirSim Python API 첫 비행 테스트
전제: 언리얼 에디터에서 Blocks(FlyingExampleMap) Play 상태
실행: python first_flight.py
성공 기준: 언리얼 화면 속 드론이 이륙 → 전진 → 사각형 비행 → 복귀 착륙
"""
import time
import cosysairsim as airsim


def main():
    client = airsim.MultirotorClient()
    print("[1/6] AirSim 연결 시도...")
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[2/6] 연결 및 Arm 완료")

    print("[3/6] 이륙...")
    client.takeoffAsync().join()
    time.sleep(1)

    # 고도 5m로 상승 (AirSim NED 좌표: z 음수 = 위)
    client.moveToZAsync(-5, 2).join()
    print("[4/6] 고도 5m 도달")

    # 사각형 경로 비행 (한 변 8m, 속도 3m/s)
    print("[5/6] 사각형 경로 비행 시작")
    waypoints = [(8, 0), (8, 8), (0, 8), (0, 0)]
    for i, (x, y) in enumerate(waypoints, 1):
        client.moveToPositionAsync(x, y, -5, 3).join()
        pos = client.getMultirotorState().kinematics_estimated.position
        print(f"    waypoint {i}/4 도달: x={pos.x_val:.1f}, y={pos.y_val:.1f}, z={pos.z_val:.1f}")
        time.sleep(0.5)

    print("[6/6] 착륙...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("완료! M1 + API 제어 검증 성공")


if __name__ == "__main__":
    main()
