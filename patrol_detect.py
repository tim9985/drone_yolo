"""
patrol_detect.py — 드론 자율 이동 + 실시간 객체 탐지 통합 루프
전제:
  - 언리얼 Play 상태
  - WSL에서 SITL + airsim-copter 실행 중 (arm + takeoff 완료 상태)
실행: python patrol_detect.py
결과:
  - 드론이 사각형 경로를 자율 비행
  - 매 프레임 YOLO가 드론 시점 영상에서 객체 탐지
  - 탐지 결과를 화면에 출력 + results/ 폴더에 저장
  - 실시간 FPS 표시
"""

import asyncio
import math
import time
import os
import threading
from collections import Counter
import cv2
import numpy as np
from pymavlink import mavutil
from ultralytics import YOLO
import cosysairsim as airsim

from map_manager import SituationMap
from coord_transform import pixel_to_ned
from tracking import MultiObjectTracker, UltralyticsTracker, boxes_from_yolo
from debug_overlay import save_debug_frame, format_coord_line
from kml_export import DetectionCollector, read_origin_from_settings

# ── 설정 ──────────────────────────────────────────────────────────────
MAVLINK_UDP   = "udp:127.0.0.1:14551" # MAVProxy 'output add 127.0.0.1:14551' 로 추가한 전용 링크
WAYPOINTS     = [                      # 사각형 경로 (x, y, z_ned)
    (15,  0, -16),
    (15, 15, -16),
    ( 0, 15, -16),
    ( 0,  0, -16),
]
SPEED         = 2.0                    # 이동 속도 (m/s)

# 비행을 부드럽게 만드는 ArduPilot 파라미터.
#
# 짐벌이 없으므로 기체가 기울면 카메라도 기운다. ANGLE_MAX 기본값은 3000
# (=30°) 이고, 실제로 이 경로에서 최대 30.5° 까지 기울었다. 그 구간의 좌표는
# 산포가 7~11m 로 못 쓴다. 측량 드론이 천천히 일정 속도로 나는 이유가 이것이다.
# 경사각을 8° 로 묶으면 대부분의 프레임이 자세 안정 조건을 만족한다.
FLIGHT_PARAMS = {
    "ANGLE_MAX":   800.0,     # 최대 경사각 (centi-deg) = 8°
    "WPNAV_SPEED": 200.0,     # 수평 순항 속도 (cm/s) = 2 m/s
    "WPNAV_ACCEL": 100.0,     # 수평 가속 (cm/s^2) — 급가속이 곧 급경사다
    "WPNAV_ACCEL_C": 100.0,   # 선회 가속
}
YOLO_MODEL    = "weights/yolov8s_stage1_all.pt"  # NOMAD(배우1~30) + WiSARD(9월·1월) 학습
YOLO_IMGSZ    = 960                    # 추론 해상도
# 기본값 0.25 는 쓰러진 사람을 버린다(재현율 0.531). 0.15 에서 0.608 로 오른다.
# 오탐은 장당 0.72 → 0.99 로 늘지만, 순간 오탐은 추적기가 3프레임 안에 폐기한다.
# 근거: threshold_tuning.csv (재현율 2배 가중 F2 가 0.10~0.15 구간에서 최대)
YOLO_CONF     = 0.15
# 탐지로 몇 번 확인돼야 지도에 신고할지. 1이면 게이트 없음(한 프레임 오탐도 신고).
#
# 2 로 두면 한 프레임만 반짝하는 오탐이 신고되지 않는다. 실비행에서 188건이 걸러졌다.
# 다만 hits 는 YOLO 가 도는 프레임에서만 오르는데(탐지 2Hz / 추적 10Hz), 그래서
# hits>=2 는 사실상 "0.5초를 살아남아라"가 된다. 고도 8m 에서 작게 찍힌 사람은
# 그 사이 광학흐름이 놓치기 쉬워 **진짜 사람까지 막힐 수 있다**
# (실측: 평균 신고 5.2명 → 2.0명. 그중 몇이 진짜였는지는 프레임별 정답이 없어 미판정).
#
# 재난 탐색에서는 놓치는 쪽이 훨씬 비싼 오류이므로, 게이트가 안전하다는 증거가
# 나오기 전까지는 끈다. 판정 방법은 정해져 있다 — AirSim 인스턴스 분할 정답으로
# 프레임별 실제 인원을 세어 게이트 유무를 비교하면 된다(auto_label.py 방식).
MIN_HITS      = 1
# 추적 백엔드: "flow"(광학흐름·자체구현) | "bytetrack" | "botsort"
# 환경변수 DRONE_TRACKER 로 덮어쓸 수 있다 — A/B 측정 시 파일을 고치지 않기 위함.
#   flow      탐지 2Hz + 추적 10Hz. YOLO 재검출 대비 약 8배 저렴
#   bytetrack 매 프레임 탐지. 낮은 신뢰도 탐지를 2차 매칭으로 회수
#   botsort   ByteTrack + 카메라 움직임 보정(GMC). 기체가 움직이는 상황에 유리할 수 있음
TRACKER       = os.environ.get("DRONE_TRACKER", "flow")
SAVE_INTERVAL = 10                     # N프레임마다 결과 이미지 저장
OUT_DIR       = "results"
MAP_DIR       = "maps"

# ── 탐지 저주기 + 추적 고주기 (작업 3) ─────────────────────────────────
DETECT_HZ     = 2.0    # YOLO 호출 주기 (Hz)
TRACK_HZ      = 10.0   # 추적 갱신 주기 (Hz)

# ── 좌표 변환 (작업 2) ────────────────────────────────────────────────
CAMERA_FOV_DEG   = 54.0   # settings.json 의 FOV_Degrees 와 일치시킬 것
# 기체 기준 카메라 피치. 실기체는 짐벌 없이 **고정 하향 90°** 로 장착한다
# (운용구조_및_예산안_v2.0 5.1). 시뮬레이터도 같은 조건이어야 하므로 -90 을 쓴다.
# 0 으로 두면 카메라가 정면을 봐서 고도를 올릴수록 지면이 화면에서 사라진다 — 실제로 겪은 문제다.
CAMERA_PITCH_DEG = -90.0

# 좌표 신고를 허용하는 기체 자세 조건.
#
# 짐벌이 없어 카메라가 기체와 함께 기운다. 기울면 화면 중앙이 지면에서
# 고도x tan(기울기) 만큼 밀리는데, 이는 자세를 알면 계산으로 상쇄된다.
# 문제는 **촬영 순간의 자세를 정확히 모른다**는 것이다. 1080p 두 장을 RPC 로
# 받는 동안 기체는 계속 기울고, 그 사이 각도 변화가 그대로 위치 오차가 된다.
#
# 실측(고도 16m, 같은 사람을 여러 번 본 결과가 한 점에 모이는 정도):
#   기울기 2°대  → 산포 0.24~0.70m
#   기울기 20°대 → 산포 6.8~10.9m
# 그래서 기울기가 크거나 촬영 중 자세가 흔들린 프레임은 탐지는 하되
# **좌표 신고에서 뺀다.** 대상은 어차피 안정 구간에서 다시 관측된다.
MAX_TILT_DEG = 6.0     # |roll|,|pitch| 합성 기울기 상한
MAX_SLEW_DEG = 1.5     # 촬영 전후 자세 변화 상한
USE_DEPTH        = True   # depth 기반 실좌표 추정 사용 (False면 기존처럼 드론 위치에 기록)

# ── 디버그 오버레이 (작업 4) ──────────────────────────────────────────
OVERLAY_INTERVAL = 10     # N프레임마다 통합 확인용 PNG 저장 (0이면 비활성)
# ──────────────────────────────────────────────────────────────────────


def connect_mavlink():
    print(f"[MAVLink] 연결 시도: {MAVLINK_UDP}")
    drone = mavutil.mavlink_connection(MAVLINK_UDP)
    drone.wait_heartbeat(timeout=10)
    print(f"[MAVLink] 연결됨 (system={drone.target_system})")
    set_flight_params(drone)
    return drone


def set_flight_params(drone, params=None):
    """경사각·속도·가속 제한을 걸어 촬영에 적합한 비행으로 만든다.

    파라미터가 실제로 반영됐는지 되읽어 확인한다. 이름이 틀리면 ArduPilot 은
    조용히 무시하므로, 확인하지 않으면 안 걸린 채로 비행하게 된다."""
    params = params or FLIGHT_PARAMS
    for name, val in params.items():
        drone.mav.param_set_send(drone.target_system, drone.target_component,
                                 name.encode(), float(val),
                                 mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(1.0)
    ok, bad = [], []
    for name, val in params.items():
        drone.mav.param_request_read_send(drone.target_system, drone.target_component,
                                          name.encode(), -1)
        m = drone.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
        if m and abs(m.param_value - val) < 1e-3:
            ok.append(f"{name}={val:g}")
        else:
            bad.append(name)
    print(f"[비행설정] 적용 {', '.join(ok)}")
    if bad:
        print(f"[비행설정] 반영 안 됨: {', '.join(bad)} — 기울기가 커져 좌표 정확도가 떨어질 수 있음")


def send_waypoint(drone, x, y, z, speed=SPEED):
    """로컬 NED 좌표로 waypoint 이동 명령"""
    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,   # position 제어만
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        0, 0
    )


def arm_and_takeoff(drone, altitude):
    """GUIDED 모드에서 arm 후 지정 고도까지 이륙 (m 단위, 양수)"""
    print("[비행] Arm 시도 (force)...")
    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0   # param2=21196: prearm 체크 강제 우회 (SITL GPS 헬스 플래핑 대응)
    )
    drone.motors_armed_wait()
    print("[비행] Armed ✓")

    print(f"[비행] Takeoff → {altitude}m")
    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )
    t_start = time.time()
    t_last_print = t_start
    while True:
        msg = drone.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=2)
        if msg and -msg.z >= altitude * 0.9:
            break
        now = time.time()
        if now - t_last_print > 3.0:
            cur_alt = -msg.z if msg else 0.0
            print(f"[비행] 상승 중... 현재고도={cur_alt:.2f}m / 목표={altitude}m "
                  f"(경과 {now - t_start:.0f}s)")
            t_last_print = now
    print(f"[비행] Takeoff 완료 ✓ (소요 {time.time()-t_start:.1f}s)")


def wait_arrival(drone, tx, ty, tz, threshold=2.0, timeout=40.0):
    """목표 지점 도착 대기 (거리 threshold m 이내).

    ArduPilot GUIDED 모드는 위치 setpoint 를 **계속 받아야** 그 목표를 유지한다.
    한 번만 보내면 수 초 뒤 실패안전이 걸려 그 자리에 정지한다(실측: 드론이
    원점에서 8m 고도로 뜬 채 사각형을 돌지 않음). 따라서 대기 중에도 4Hz 로
    재전송한다.

    반환: (도달 여부, 마지막 거리)
    """
    t0 = time.time()
    last_send = 0.0
    dist = float("inf")
    while time.time() - t0 < timeout:
        if time.time() - last_send >= 0.25:
            send_waypoint(drone, tx, ty, tz)
            last_send = time.time()
        msg = drone.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=0.3)
        if msg:
            dist = ((msg.x - tx) ** 2 + (msg.y - ty) ** 2 + (msg.z - tz) ** 2) ** 0.5
            if dist < threshold:
                return True, dist
    return False, dist


# ── 탐지 스레드 ────────────────────────────────────────────────────────
class DetectionThread(threading.Thread):
    """탐지 저주기 + 추적 고주기 구조.

    detect_hz : YOLO 호출 주기 (기본 2Hz)
    track_hz  : 추적 갱신 주기 (기본 10Hz)
    매 프레임 YOLO를 돌리지 않으므로 GPU 부하가 줄고, 탐지 사이 구간은
    OpenCV 추적기 + 칼만 필터가 위치를 이어받는다.
    """

    def __init__(self, model, sit_map, detect_hz=DETECT_HZ, track_hz=TRACK_HZ,
                 use_depth=USE_DEPTH, overlay_interval=OVERLAY_INTERVAL):
        super().__init__(daemon=True)
        self.client  = None  # 스레드 내부에서 생성 (msgpackrpc는 스레드별 이벤트 루프 필요)
        self.model   = model
        self.sit_map = sit_map
        self.running = True
        self.frame   = 0
        self.fps     = 0.0
        self.detect_period = 1.0 / max(detect_hz, 1e-6)
        self.track_period  = 1.0 / max(track_hz, 1e-6)
        self.use_depth = use_depth
        self.overlay_interval = overlay_interval
        # 백엔드 선택 — 어느 쪽이든 Track 객체를 돌려주므로 이후 코드는 동일하다
        if TRACKER in UltralyticsTracker.CONFIGS:
            self.mot = UltralyticsTracker(model, backend=TRACKER,
                                          imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
        else:
            self.mot = MultiObjectTracker()
        self.n_detect_frames = 0   # YOLO를 실제로 호출한 프레임 수
        self.n_track_frames  = 0   # 추적만 한 프레임 수
        self.n_gated         = 0   # 게이트에서 걸러낸 트랙 보고 건수
        self.n_unstable      = 0   # 자세 불안정으로 좌표 신고를 건너뛴 프레임 수
        self.pose_stable     = True
        self.tilt_deg        = 0.0
        self.att_slew_deg    = 0.0
        # 프레임 단위 탐지를 사람 단위 신고로 묶어 KML(지도)로 내보낸다
        origin, origin_alt = read_origin_from_settings()
        self.kml = DetectionCollector(origin_gps=origin, origin_alt_m=origin_alt)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(MAP_DIR, exist_ok=True)

    def _capture(self):
        """RGB (+옵션 depth) 캡처."""
        reqs = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        if self.use_depth:
            reqs.append(airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True))
        resp = self.client.simGetImages(reqs)
        if not resp or resp[0].width == 0:
            return None, None
        r = resp[0]
        img = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
        depth = None
        if self.use_depth and len(resp) > 1:
            d = resp[1]
            if d.width:
                depth = np.array(d.image_data_float, dtype=np.float32).reshape(d.height, d.width)
        return img, depth

    def _world_from_track(self, t, depth, img_shape, drone_ned, attitude):
        """추적 박스 중심 + depth → 로컬 NED 실좌표. depth가 없으면 None."""
        if depth is None:
            return None
        h, w = img_shape[:2]
        cx, cy = t.center
        # depth 해상도가 다르면 비례 변환
        dy = int(round(cy * depth.shape[0] / h))
        dx = int(round(cx * depth.shape[1] / w))
        dy = min(max(dy, 0), depth.shape[0] - 1)
        dx = min(max(dx, 0), depth.shape[1] - 1)
        d = float(depth[dy, dx])
        if not np.isfinite(d) or d <= 0 or d > 500:
            return None
        return pixel_to_ned(cx, cy, w, h, CAMERA_FOV_DEG, drone_ned, attitude, d,
                            camera_orientation=(0.0, math.radians(CAMERA_PITCH_DEG), 0.0),
                            depth_type="planar")

    def run(self):
        # 서브스레드에는 asyncio 이벤트 루프가 없어 msgpackrpc(cosysairsim 내부)가 실패함
        # → 스레드 전용 이벤트 루프 생성 + AirSim 클라이언트도 스레드 내부에서 별도 생성
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # 고정 하향 장착을 시뮬레이터에 반영 (짐벌 없음 — 기체와 함께 기운다)
        self.client.simSetCameraPose(
            "0", airsim.Pose(airsim.Vector3r(0, 0, 0),
                             airsim.utils.euler_to_quaternion(
                                 0.0, math.radians(CAMERA_PITCH_DEG), 0.0)))
        print(f"[카메라] 고정 하향 {CAMERA_PITCH_DEG}° 적용")

        t_prev = time.time()
        t_last_detect = 0.0
        annotated = None
        while self.running:
            try:
                loop_t0 = time.time()
                # 촬영 **직전** 자세를 먼저 잡는다.
                # 1080p 두 장을 RPC 로 받는 데 수백 ms 가 걸리고, 그 사이 기체는 계속 기운다.
                # 촬영 뒤에만 읽으면 셔터 시점과 자세가 어긋나 보정이 상쇄되지 않는다.
                # (실측: 기울기 10° 이상에서 위치 오차 중앙 3.07m, 이론 밀림값과 거의 같았다)
                kin_pre = self.client.simGetGroundTruthKinematics()
                img, depth = self._capture()
                if img is None:
                    time.sleep(0.05)
                    continue

                # 드론 현재 위치·자세 조회 (로컬 NED) → 좌표 변환·지도 갱신에 사용
                #
                # getMultirotorState().kinematics_estimated 는 VehicleType=ArduCopter
                # (외부 펌웨어 SITL) 에서 (0,0,0) 만 돌려준다. 상태 추정을 AirSim 이 아니라
                # ArduPilot 이 하고 있어서 AirSim 쪽 추정값이 채워지지 않는다.
                # simGetGroundTruthKinematics 는 시뮬레이터가 직접 아는 참값이라 항상 유효하다.
                kin = self.client.simGetGroundTruthKinematics()
                pos = kin.position
                if pos.x_val == 0.0 and pos.y_val == 0.0 and pos.z_val == 0.0:
                    # 참값도 0이면 아직 스폰 전이거나 연결 문제 — 추정값으로 재시도
                    kin = self.client.getMultirotorState().kinematics_estimated
                    pos = kin.position
                drone_ned = (pos.x_val, pos.y_val, pos.z_val)
                drone_x, drone_y, drone_alt = pos.x_val, pos.y_val, -pos.z_val
                # Cosys-AirSim 은 레거시 AirSim 의 to_eularian_angles 를 제공하지 않는다.
                # 이름만 다른 게 아니라 **반환 순서도 다르다**:
                #   레거시 to_eularian_angles      → (pitch, roll, yaw)
                #   Cosys  quaternion_to_euler_angles → (roll, pitch, yaw)
                # 따라서 예전처럼 [1],[0] 로 뒤바꾸면 roll 과 pitch 가 서로 바뀐다.
                att_post = airsim.quaternion_to_euler_angles(kin.orientation)  # (roll, pitch, yaw)
                att_pre = airsim.quaternion_to_euler_angles(kin_pre.orientation)
                # 셔터는 두 시점 사이 어딘가다 → 중간값이 최선의 추정이다.
                # yaw 는 ±π 경계를 넘을 수 있어 각도 차를 감아서 더한다.
                def _mid(a, b):
                    return a + math.atan2(math.sin(b - a), math.cos(b - a)) / 2.0
                att = tuple(_mid(a, b) for a, b in zip(att_pre, att_post))
                # 촬영 구간 동안 자세가 얼마나 흔들렸는지 — 클수록 그 프레임의 좌표는 못 믿는다
                self.att_slew_deg = math.degrees(
                    math.hypot(att_post[0] - att_pre[0], att_post[1] - att_pre[1]))
                self.tilt_deg = math.degrees(math.hypot(att[0], att[1]))
                self.pose_stable = (self.tilt_deg <= MAX_TILT_DEG
                                    and self.att_slew_deg <= MAX_SLEW_DEG)
                if not self.pose_stable:
                    self.n_unstable += 1
                pos_pre = kin_pre.position
                drone_ned = tuple((a + b) / 2.0 for a, b in
                                  zip(drone_ned, (pos_pre.x_val, pos_pre.y_val, pos_pre.z_val)))
                drone_x, drone_y, drone_alt = drone_ned[0], drone_ned[1], -drone_ned[2]
                self.kml.add_drone(time.time(), drone_x, drone_y, drone_alt)

                # ── 추적 백엔드별 갱신 ──
                if isinstance(self.mot, UltralyticsTracker):
                    # ByteTrack·BoT-SORT 는 매 프레임 탐지를 전제한다
                    tracks = self.mot.step(img, self.frame)
                    dets = self.mot.last_dets
                    do_detect = True
                    annotated = self.mot.last_plot
                    self.n_detect_frames += 1
                else:
                    # 탐지 저주기 / 추적 고주기 (광학흐름 백엔드)
                    do_detect = (loop_t0 - t_last_detect) >= self.detect_period
                    if do_detect:
                        results = self.model(img, verbose=False, imgsz=YOLO_IMGSZ, conf=YOLO_CONF)
                        dets = boxes_from_yolo(results[0], self.model.names)
                        tracks = self.mot.update_with_detections(img, dets, self.frame)
                        annotated = results[0].plot()
                        t_last_detect = loop_t0
                        self.n_detect_frames += 1
                    else:
                        dets = []
                        tracks = self.mot.update_tracking_only(img, self.frame)
                        self.n_track_frames += 1

                # ── 신고 게이트: 탐지로 2회 이상 확인된 트랙만 지도에 올린다 ──
                #
                # 추적기는 3프레임 연속 실패한 트랙을 폐기하지만, 그 전에 이미 한 번은
                # 신고된다. 바위 그림자처럼 한 프레임만 반짝하는 오탐이 그렇게 새어 나간다.
                # (실측: 사람 없는 배경 289장에서 장당 0.99건)
                # hits >= 2 를 걸면 그런 것들은 아예 신고되지 않는다. 대가는 진짜 사람의
                # 신고가 탐지 1주기(2Hz 기준 약 0.5초) 늦어지는 것뿐이다.
                reported = [t for t in tracks if t.hits >= MIN_HITS]
                self.n_gated += len(tracks) - len(reported)

                # ── 추적 객체별 실좌표 추정 → 8층 지도 입력 ──
                detections = []
                for t in reported:
                    entry = {"class": t.cls_name, "confidence": t.confidence,
                             "track_id": t.track_id}
                    pt = self._world_from_track(t, depth, img.shape, drone_ned, att)
                    if pt is not None:
                        # map_manager 가 쓰는 드론 기준 상대 좌표(m)
                        entry["rel_x"] = pt.x - drone_x
                        entry["rel_y"] = pt.y - drone_y
                        entry["world_x"], entry["world_y"] = pt.x, pt.y
                        # HEIGHT 층 입력: 대상 상단의 높이(m).
                        # NED z 는 아래가 +이고 이륙 지점(z=0)을 지면으로 보므로 -z 가 높이.
                        entry["height_m"] = max(0.0, -pt.z)
                        # 지도 신고는 **YOLO 가 실제로 확인한 프레임에서만** 한다.
                        # 추적 전용 프레임의 상자는 광학흐름으로 밀어 놓은 추정치라
                        # 조금씩 표류하고, 그 상자 중심의 깊이는 엉뚱한 배경을 찍는다.
                        # 그대로 신고했더니 위치 산포가 7.6m 까지 벌어졌다 (16m 고도 실측).
                        # 추적기의 역할은 탐지 사이의 ID 연속성이지 위치 측정이 아니다.
                        if do_detect and self.pose_stable:
                            self.kml.add(time.time(), t.cls_name, t.confidence,
                                         pt.x, pt.y, entry["height_m"],
                                         track_id=t.track_id,
                                         roll_deg=math.degrees(att[0]),
                                         pitch_deg=math.degrees(att[1]))
                    detections.append(entry)

                self.sit_map.step(dt=1.0)
                self.sit_map.update_from_detection(detections, drone_x, drone_y, drone_alt)

                # FPS 계산
                now       = time.time()
                self.fps  = 1.0 / max(now - t_prev, 1e-6)
                t_prev    = now

                # 콘솔 출력 (클래스별 개수 포함)
                if detections:
                    counts = Counter(d["class"] for d in detections)
                    cls_str = ", ".join(f"{cls} {cnt}" for cls, cnt in counts.items())
                else:
                    cls_str = "탐지없음"
                print(f"[{'탐지' if do_detect else '추적'}] 프레임={self.frame:04d} | {cls_str} | "
                      f"FPS={self.fps:.1f} | 위치=({drone_x:.1f},{drone_y:.1f},{drone_alt:.1f}) | "
                      f"미관측 {self.sit_map.get_unobserved_ratio()*100:.0f}%")

                # 결과 이미지 저장 (N프레임마다)
                if self.frame % SAVE_INTERVAL == 0 and annotated is not None:
                    cv2.imwrite(f"{OUT_DIR}/frame_{self.frame:04d}.png", annotated)
                    self.sit_map.save(f"{MAP_DIR}/map_{self.frame:04d}.png")

                # 통합 확인용 오버레이 (탐지·추적·Depth·좌표를 한 장에)
                if self.overlay_interval and self.frame % self.overlay_interval == 0:
                    est = next((( d["world_x"], d["world_y"]) for d in detections
                                if "world_x" in d), None)
                    lines = [format_coord_line(est_ned=est)] if est else []
                    lines.append(f"det {self.n_detect_frames} / trk {self.n_track_frames} "
                                 f"| backend {self.mot.backend} | FPS {self.fps:.1f}")
                    save_debug_frame(img, self.frame, detections=[
                        {"class": d["class"], "confidence": d["confidence"],
                         "bbox": t.bbox} for d, t in zip(detections, reported)] if do_detect else [],
                        tracks=reported, depth=depth, info_lines=lines)

                self.frame += 1

                # 추적 주기 유지 (과도한 폴링 방지)
                sleep_left = self.track_period - (time.time() - loop_t0)
                if sleep_left > 0:
                    time.sleep(sleep_left)

            except Exception as e:
                print(f"[탐지 에러] {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False


# ── 재사용 가능한 정찰 루프 ────────────────────────────────────────────
def run_patrol(drone, ac_client=None):
    """이미 연결·Arm·GUIDED·이륙까지 완료된 drone(MAVLink 연결)을 받아
    사각형 경로 정찰 + 탐지 루프를 실행하고 RTL까지 수행한다.
    run_experiment.py 등 오케스트레이터가 자체 MAVLink 연결을 재사용할 때 호출."""
    # 오케스트레이터는 자체 연결 함수를 쓰므로 connect_mavlink 를 거치지 않는다.
    # 비행 파라미터는 어느 경로로 들어오든 걸려야 하니 여기서 한 번 더 적용한다.
    set_flight_params(drone)
    if ac_client is None:
        print("[AirSim] 연결 중...")
        ac_client = airsim.MultirotorClient()
        ac_client.confirmConnection()
        print("[AirSim] 연결됨")

    print(f"[YOLO] 모델 로드: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    # 8층 지도 초기화
    sit_map = SituationMap(width=40, height=40, cell_size=1.0, origin_offset=(10.0, 10.0))

    # 탐지 스레드 시작
    det = DetectionThread(model, sit_map)
    det.start()
    print("[탐지] 스레드 시작")

    # 사각형 경로 자율 비행
    print(f"\n[비행] 사각형 경로 시작 ({len(WAYPOINTS)}개 waypoint)")
    for i, (x, y, z) in enumerate(WAYPOINTS, 1):
        print(f"\n[비행] waypoint {i}/{len(WAYPOINTS)}: x={x}, y={y}, z={z}")
        send_waypoint(drone, x, y, z)
        ok, dist = wait_arrival(drone, x, y, z, threshold=2.0)
        mark = "도달 ✓" if ok else f"미도달 ✗ (남은 거리 {dist:.1f}m)"
        print(f"[비행] waypoint {i} {mark}  | 탐지 FPS: {det.fps:.1f}")
        time.sleep(0.5)

    # 복귀
    print("\n[비행] 사각형 완료. 복귀(RTL)...")
    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(10)

    # 종료
    det.stop()
    sit_map.save(f"{MAP_DIR}/map_final.png")
    print(f"\n[완료] 총 {det.frame}프레임 처리 | 결과: {OUT_DIR}/ | 지도: {MAP_DIR}/map_final.png")
    print(f"[지도] 최종 미관측 비율: {sit_map.get_unobserved_ratio()*100:.0f}%")
    kml_path = det.kml.write_kml(f"{OUT_DIR}/detections.kml",
                                 note=f"{det.frame}프레임 · 모델 {YOLO_MODEL} · 임계값 {YOLO_CONF}")
    csv_path = det.kml.write_raw_csv(f"{OUT_DIR}/detections_raw.csv")
    print(f"[원본] {csv_path}  병합 전 탐지 {len(det.kml.raw)}건 "
          f"(merge_tune.py 로 반경 재분석 가능)")
    ks = det.kml.summary()
    print(f"[KML] {kml_path}  확인 대상 {ks['targets_confirmed']}명 / "
          f"미확인 후보 {ks['targets_candidate']}건 (탐지 {ks['detections_total']}건에서 병합)")
    print(f"[자세] 기울기>{MAX_TILT_DEG}° 또는 촬영중 흔들림>{MAX_SLEW_DEG}° 로 "
          f"좌표 신고를 건너뛴 프레임 {det.n_unstable}/{det.frame}")
    print(f"[게이트] hits<{MIN_HITS} 로 신고 보류된 트랙 {det.n_gated}건 "
          f"(한 프레임만 반짝한 오탐 후보)")
    print("M4 + M5 + M6 + 8층 지도(STEP 10) 통합 루프 달성!")
    return {"frames": det.frame, "unobserved_ratio": sit_map.get_unobserved_ratio()}


# ── 독립 실행 진입점 ──────────────────────────────────────────────────
def main():
    """단독 실행(python patrol_detect.py)용: 자체적으로 연결·Arm·이륙까지 수행."""
    print("[AirSim] 연결 중...")
    ac = airsim.MultirotorClient()
    ac.confirmConnection()
    print("[AirSim] 연결됨")

    drone = connect_mavlink()

    # GUIDED 모드 확인
    drone.mav.command_long_send(
        drone.target_system, drone.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 1, 4, 0, 0, 0, 0, 0   # GUIDED = 4
    )
    time.sleep(1)

    # Arm + Takeoff (첫 waypoint 고도까지)
    arm_and_takeoff(drone, altitude=-WAYPOINTS[0][2])

    run_patrol(drone, ac)


if __name__ == "__main__":
    main()
