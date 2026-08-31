"""
coord_transform.py — 픽셀 좌표 → 실세계 좌표 변환 (작업 2)

탐지 박스의 픽셀 좌표를 드론의 GPS·고도·자세와 Depth(ground truth)를 이용해
실세계 좌표로 환산한다. 8층 지도(map_manager)에 기록될 입력이 된다.

좌표계 규약
  · NED : x=North, y=East, z=Down (AirSim/MAVLink 로컬 좌표와 동일)
  · 기체(body) : x=전방, y=우측, z=하방
  · 카메라 : 자세가 (0,0,0)이면 기체와 축이 같다(전방 주시).
             수직 하방 촬영은 camera_orientation=(0, -pi/2, 0).
  · 영상 : u(가로)는 우측 +, v(세로)는 하단 +

Depth 종류 주의
  · AirSim ImageType.DepthPlanar = 광축(카메라 +X) 방향 성분 거리 (Z-buffer식)
  · AirSim ImageType.DepthPerspective = 광선 방향 유클리드 거리
  collect_data.py 는 DepthPlanar 를 저장하므로 기본값을 planar 로 둔다.
  이 구분을 틀리면 화면 가장자리 객체에서 체계적 오차가 생긴다.

주요 함수
  pixel_to_ned()   : 픽셀 → 로컬 NED 좌표 (map_manager 가 그대로 쓰는 형식)
  pixel_to_world() : 픽셀 → (위도, 경도)  ※ 과제 지정 시그니처
"""
import math
from collections import namedtuple

# 위경도 근사 변환 상수 (WGS84 기준, 소규모 영역에서 충분)
M_PER_DEG_LAT = 111320.0

NedPoint = namedtuple("NedPoint", ["x", "y", "z", "range_m"])


# ── 회전 행렬 (3-2-1 항공 규약) ──────────────────────────────────────────
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _matvec(A, v):
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def rotation_body_to_ned(roll, pitch, yaw):
    """기체 자세(rad) → body→NED 회전행렬. Rz(yaw)·Ry(pitch)·Rx(roll)."""
    return _matmul(_matmul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


# ── 핵심 변환 ───────────────────────────────────────────────────────────
def pixel_ray_camera(px, py, image_width, image_height, fov_deg):
    """픽셀 → 카메라 좌표계 방향벡터 (정규화 안 함, x=1 기준).
    fov_deg 는 수평 화각. 정사각 픽셀 가정으로 fy=fx."""
    cx, cy = image_width / 2.0, image_height / 2.0
    fx = (image_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    return (1.0, (px - cx) / fx, (py - cy) / fy)


def pixel_to_ned(px, py, image_width, image_height, fov_deg,
                 drone_ned, drone_attitude, depth_value,
                 camera_orientation=(0.0, 0.0, 0.0),
                 depth_type="planar",
                 apply_attitude=True):
    """픽셀 + depth → 로컬 NED 좌표.

    drone_ned        : (x, y, z) 드론 위치 (m, NED. z는 아래가 +)
    drone_attitude   : (roll, pitch, yaw) 라디안
    camera_orientation: 기체 기준 카메라 자세 (roll, pitch, yaw) 라디안
                        수직 하방이면 (0, -pi/2, 0)
    depth_value      : 해당 픽셀의 depth (m)
    apply_attitude   : False 면 기체 자세를 무시(수직 하방 가정만 적용).
                       자세 보정 효과를 비교 측정하기 위한 스위치.
    반환: NedPoint(x, y, z, range_m)
    """
    ray_cam = pixel_ray_camera(px, py, image_width, image_height, fov_deg)

    if depth_type == "planar":
        # planar depth는 광축 성분 → 비정규화 광선에 그대로 곱하면 된다
        pt_cam = tuple(depth_value * c for c in ray_cam)
    else:  # perspective(유클리드)
        n = math.sqrt(sum(c * c for c in ray_cam))
        pt_cam = tuple(depth_value * c / n for c in ray_cam)

    # 카메라 → 기체
    r_c, p_c, y_c = camera_orientation
    pt_body = _matvec(rotation_body_to_ned(r_c, p_c, y_c), pt_cam)

    # 기체 → NED
    if apply_attitude:
        roll, pitch, yaw = drone_attitude
    else:
        # 자세 미보정 비교군: 롤·피치를 0으로 두고 방위(yaw)만 사용
        roll, pitch, yaw = 0.0, 0.0, drone_attitude[2]
    pt_ned_rel = _matvec(rotation_body_to_ned(roll, pitch, yaw), pt_body)

    x = drone_ned[0] + pt_ned_rel[0]
    y = drone_ned[1] + pt_ned_rel[1]
    z = drone_ned[2] + pt_ned_rel[2]
    rng = math.sqrt(sum(c * c for c in pt_ned_rel))
    return NedPoint(x, y, z, rng)


def ned_to_gps(origin_gps, north_m, east_m):
    """로컬 NED 오프셋(m) → (위도, 경도). 소규모 영역 근사식."""
    lat0, lon0 = origin_gps
    dlat = north_m / M_PER_DEG_LAT
    dlon = east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def gps_to_ned(origin_gps, lat, lon):
    """(위도, 경도) → 원점 기준 NED 오프셋 (north_m, east_m)."""
    lat0, lon0 = origin_gps
    north = (lat - lat0) * M_PER_DEG_LAT
    east = (lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return north, east


def pixel_to_world(px, py, image_width, image_height, fov_deg,
                   drone_gps, drone_altitude, drone_attitude, depth_value,
                   camera_orientation=(0.0, 0.0, 0.0),
                   depth_type="planar",
                   apply_attitude=True):
    """과제 지정 시그니처. 픽셀 → (위도, 경도).

    drone_gps      : (lat, lon) 드론 현재 위치
    drone_altitude : 고도 (m, 양수)
    나머지 인자는 pixel_to_ned 와 동일.

    드론 GPS를 로컬 원점으로 두고 상대 NED를 구한 뒤 위경도로 환산한다.
    """
    pt = pixel_to_ned(px, py, image_width, image_height, fov_deg,
                      drone_ned=(0.0, 0.0, -float(drone_altitude)),
                      drone_attitude=drone_attitude,
                      depth_value=depth_value,
                      camera_orientation=camera_orientation,
                      depth_type=depth_type,
                      apply_attitude=apply_attitude)
    return ned_to_gps(drone_gps, pt.x, pt.y)


# ── 자체 점검 ───────────────────────────────────────────────────────────
def _selftest():
    """기하가 맞는지 손으로 검산 가능한 케이스 몇 개."""
    W, H, FOV = 1280, 720, 90.0
    ok = True

    # 1) 수직 하방, 자세 수평, 화면 중심 → 드론 바로 아래 지면
    pt = pixel_to_ned(W / 2, H / 2, W, H, FOV,
                      drone_ned=(10.0, 5.0, -20.0), drone_attitude=(0, 0, 0),
                      depth_value=20.0, camera_orientation=(0, -math.pi / 2, 0))
    ok &= abs(pt.x - 10.0) < 1e-6 and abs(pt.y - 5.0) < 1e-6 and abs(pt.z - 0.0) < 1e-6
    print(f"  중심 픽셀 → ({pt.x:.3f}, {pt.y:.3f}, {pt.z:.3f})  기대 (10, 5, 0)")

    # 2) 수직 하방, 우측 가장자리 → 동쪽으로 고도*tan(FOV/2) 만큼 이동
    expect_e = 20.0 * math.tan(math.radians(FOV / 2))
    pt = pixel_to_ned(W, H / 2, W, H, FOV,
                      drone_ned=(0.0, 0.0, -20.0), drone_attitude=(0, 0, 0),
                      depth_value=20.0, camera_orientation=(0, -math.pi / 2, 0))
    ok &= abs(pt.y - expect_e) < 1e-6
    print(f"  우측끝 픽셀 → y={pt.y:.3f}  기대 {expect_e:.3f}")

    # 3) yaw 90도(동쪽 주시)에서 전방 카메라 중심 → 동쪽으로 depth 만큼
    pt = pixel_to_ned(W / 2, H / 2, W, H, FOV,
                      drone_ned=(0.0, 0.0, -10.0), drone_attitude=(0, 0, math.pi / 2),
                      depth_value=30.0, camera_orientation=(0, 0, 0))
    ok &= abs(pt.x) < 1e-6 and abs(pt.y - 30.0) < 1e-6
    print(f"  yaw90 전방 → ({pt.x:.3f}, {pt.y:.3f})  기대 (0, 30)")

    # 4) GPS 왕복 변환
    origin = (35.1796, 129.0756)
    lat, lon = ned_to_gps(origin, 100.0, -50.0)
    n, e = gps_to_ned(origin, lat, lon)
    ok &= abs(n - 100.0) < 1e-6 and abs(e + 50.0) < 1e-6
    print(f"  GPS 왕복 → north={n:.6f}, east={e:.6f}  기대 (100, -50)")

    print("자체 점검:", "통과" if ok else "실패")
    return ok


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _selftest()
