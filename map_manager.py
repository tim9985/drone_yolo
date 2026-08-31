"""
map_manager.py — 전역 2.5차원 8층 지도 (SituationMap)

좌표계: AirSim/MAVLink 로컬 NED (x=north, y=east, 단위 m) 를 그대로 사용.
        GPS 위경도가 필요해지면 gps_to_cell()을 추가하면 됨 (현재는 local_to_cell만 사용).
"""
import numpy as np

# 클래스명은 사용하는 가중치에 따라 다르다.
#   COCO(yolov8n)     : person / car, truck, bus, motorcycle, bicycle
#   VisDrone(yolov8s) : pedestrian, people / bicycle, car, van, truck,
#                       tricycle, awning-tricycle, bus, motor
# 두 체계를 모두 받도록 합집합으로 둔다. (VisDrone 가중치로 교체한 뒤
#  'pedestrian' 이 person 으로 매칭되지 않아 PERSON_PROB/TARGET_BELIEF 층이
#  전혀 갱신되지 않던 문제를 수정)
PERSON_CLASSES = {"person", "pedestrian", "people"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle",
                   "van", "motor", "tricycle", "awning-tricycle"}


class SituationMap:
    # 층 인덱스
    OCCUPANCY    = 0   # 0=미관측, 0.5=빈공간, 1=점유
    HEIGHT       = 1   # 장애물 최고 높이 (m)
    PERSON_PROB  = 2   # 사람 존재 확률
    VEHICLE_PROB = 3   # 차량 존재 확률
    CONFIDENCE   = 4   # 인식 신뢰도 (최근 관측 기준)
    LAST_OBS     = 5   # 마지막 관측 이후 경과 시간 (s)
    RISK         = 6   # 위험도 (동적 객체 근접 시 상승)
    TARGET_BELIEF = 7  # 대상 존재 신념 (현재는 사람 확률과 연동, 추적 임무 확장 지점)

    def __init__(self, width=40, height=40, cell_size=1.0, origin_offset=(10.0, 10.0)):
        """
        width/height : 격자 셀 수
        cell_size    : 셀 한 변의 실제 크기 (m)
        origin_offset: 로컬 NED (0,0)이 격자의 어느 셀에 대응하는지 (m 단위 오프셋)
                        예: (10,10) → NED (-10,-10) ~ (30,30) 범위를 격자가 커버
        """
        self.grid = np.zeros((8, height, width), dtype=np.float32)
        self.cell_size = cell_size
        self.origin_offset = origin_offset
        self.width = width
        self.height = height
        self._t = 0.0  # 내부 스텝 카운터 (LAST_OBS 갱신용, 실제 시계 대신 프레임 기준)

    def local_to_cell(self, x, y):
        """로컬 NED (x=north, y=east, m) → 격자 셀 인덱스 (cx, cy)"""
        cx = int((y + self.origin_offset[1]) / self.cell_size)
        cy = int((x + self.origin_offset[0]) / self.cell_size)
        return int(np.clip(cx, 0, self.width - 1)), int(np.clip(cy, 0, self.height - 1))

    def step(self, dt=1.0):
        """모든 셀의 LAST_OBS(경과시간) 증가. 매 프레임 한 번 호출."""
        self._t += dt
        mask = self.grid[self.OCCUPANCY] > 0
        self.grid[self.LAST_OBS][mask] += dt

    def update_from_detection(self, detections, drone_x, drone_y, drone_alt, fov_margin=1.5):
        """
        detections: [{"class": str, "confidence": float, "rel_x": float, "rel_y": float}, ...]
                    rel_x/rel_y 는 드론 기준 상대 위치(m). 없으면 드론 위치에 그대로 기록.
        drone_x, drone_y, drone_alt: 드론 현재 로컬 NED 위치 (m), alt는 양수(고도)
        """
        cx, cy = self.local_to_cell(drone_x, drone_y)

        # 관측 반경: 고도가 높을수록 더 넓게 관측했다고 가정 (간단화된 FOV 모델)
        fov_radius_m = max(drone_alt * fov_margin * 0.5, self.cell_size)
        fov_radius_cells = max(int(fov_radius_m / self.cell_size), 1)

        for dy in range(-fov_radius_cells, fov_radius_cells + 1):
            for dx in range(-fov_radius_cells, fov_radius_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[self.OCCUPANCY, ny, nx] == 0:
                        self.grid[self.OCCUPANCY, ny, nx] = 0.5  # 빈 공간으로 갱신
                    self.grid[self.LAST_OBS, ny, nx] = 0.0

        for det in detections:
            cls = det.get("class", "")
            conf = float(det.get("confidence", 0.0))
            # coord_transform 이 추정한 상대 좌표가 있으면 실제 위치 셀에 기록한다.
            # (없으면 종전대로 드론 위치 셀에 기록 — depth 미사용/추정 실패 시)
            if "rel_x" in det and "rel_y" in det:
                ox, oy = self.local_to_cell(drone_x + float(det["rel_x"]),
                                            drone_y + float(det["rel_y"]))
            else:
                ox, oy = cx, cy

            self.grid[self.CONFIDENCE, oy, ox] = max(self.grid[self.CONFIDENCE, oy, ox], conf)
            self.grid[self.OCCUPANCY, oy, ox] = 1.0

            # HEIGHT 층: depth 로 추정한 대상 높이(m). coord_transform 을 쓰는 호출부가
            # height_m 을 넣어주면 기록된다(없으면 종전대로 비워 둠).
            if "height_m" in det:
                h = float(det["height_m"])
                if np.isfinite(h):
                    self.grid[self.HEIGHT, oy, ox] = max(self.grid[self.HEIGHT, oy, ox], h)

            if cls in PERSON_CLASSES:
                self.grid[self.PERSON_PROB, oy, ox] = max(self.grid[self.PERSON_PROB, oy, ox], conf)
                self.grid[self.TARGET_BELIEF, oy, ox] = max(self.grid[self.TARGET_BELIEF, oy, ox], conf)
                self.grid[self.RISK, oy, ox] = min(self.grid[self.RISK, oy, ox] + 0.1, 1.0)
            elif cls in VEHICLE_CLASSES:
                self.grid[self.VEHICLE_PROB, oy, ox] = max(self.grid[self.VEHICLE_PROB, oy, ox], conf)
                self.grid[self.RISK, oy, ox] = min(self.grid[self.RISK, oy, ox] + 0.05, 1.0)

    def get_unobserved_ratio(self):
        unobs = np.sum(self.grid[self.OCCUPANCY] == 0)
        total = self.width * self.height
        return float(unobs) / total

    def visualize(self, scale=12):
        """8층 중 핵심 정보를 하나의 BGR 이미지로 합성 (results/에 저장용)"""
        import cv2

        vis = np.full((self.height, self.width, 3), (50, 50, 50), dtype=np.uint8)  # 미관측: 회색
        vis[self.grid[self.OCCUPANCY] == 0.5] = (200, 200, 200)  # 빈 공간: 연회색
        vis[self.grid[self.OCCUPANCY] == 1.0] = (120, 120, 255)  # 점유(미분류): 옅은 빨강

        person_mask = self.grid[self.PERSON_PROB] > 0.3
        vis[person_mask] = (0, 255, 0)       # 사람: 초록
        vehicle_mask = self.grid[self.VEHICLE_PROB] > 0.3
        vis[vehicle_mask] = (0, 165, 255)    # 차량: 주황

        out = cv2.resize(vis, (self.width * scale, self.height * scale),
                          interpolation=cv2.INTER_NEAREST)
        return out

    def save(self, path):
        import cv2
        cv2.imwrite(path, self.visualize())
