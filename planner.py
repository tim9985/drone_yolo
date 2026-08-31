"""
planner.py — Next-Best-View 후보 생성 + 규칙 기반 선택 (STEP 11)

8층 지도(SituationMap)를 입력받아 다음 관측 지점을 결정한다.
후보 생성은 알고리즘(Frontier·의미 기반·재관측), 선택은 규칙 기반 가중 점수로 수행.
→ 추후 "학습된 후보 평가 모델"로 select_next() 부분만 교체하면 되는 구조.
"""
import numpy as np


class NextBestViewPlanner:
    def __init__(self, situation_map, cruise_alt=8.0, search_radius_cells=15):
        self.map = situation_map
        self.cruise_alt = cruise_alt
        self.search_radius_cells = search_radius_cells

    def cell_to_local(self, cx, cy):
        """격자 셀 인덱스 → 로컬 NED (x, y) 셀 중심 좌표 (m)"""
        y = (cx + 0.5) * self.map.cell_size - self.map.origin_offset[1]
        x = (cy + 0.5) * self.map.cell_size - self.map.origin_offset[0]
        return x, y

    def observation_radius_cells(self):
        """map_manager 가 한 번의 관측으로 '관측됨'으로 칠하는 반경(셀).
        SituationMap.update_from_detection 의 FOV 모델과 동일한 식을 쓴다."""
        fov_radius_m = max(self.cruise_alt * 1.5 * 0.5, self.map.cell_size)
        return max(int(fov_radius_m / self.map.cell_size), 1)

    def effective_search_radius(self):
        """탐색 반경은 관측 반경보다 커야 한다.

        관측 반경 >= 탐색 반경이면 탐색창 안이 전부 '관측됨'으로 채워져
        frontier(미관측·관측 경계) 후보가 하나도 잡히지 않는다. 실제로 고도 20m
        에서는 관측 15셀 = 기본 탐색 15셀이라 후보 0개 → next_waypoint()가 None을
        반환하고 자율 정찰이 즉시 종료된다. 관측 반경 + 여유만큼 자동 확장한다.
        """
        return max(self.search_radius_cells, self.observation_radius_cells() + 3)

    def generate_candidates(self, drone_x, drone_y):
        grid = self.map.grid
        cx, cy = self.map.local_to_cell(drone_x, drone_y)
        r = self.effective_search_radius()
        candidates = []

        # 1. Frontier 후보 — 관측/미관측 경계
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.map.width and 0 <= ny < self.map.height):
                    continue
                if grid[self.map.OCCUPANCY, ny, nx] != 0:
                    continue
                for ddy, ddx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nnx, nny = nx + ddx, ny + ddy
                    if 0 <= nnx < self.map.width and 0 <= nny < self.map.height:
                        if grid[self.map.OCCUPANCY, nny, nnx] > 0:
                            candidates.append({"cell": (nx, ny), "type": "frontier", "score": 1.0})
                            break

        # 2. 의미 기반 후보 — 사람 확률이 높은 곳 (상세관측/추적 후보)
        person_cells = np.argwhere(grid[self.map.PERSON_PROB] > 0.3)
        for ny, nx in person_cells:
            candidates.append({
                "cell": (int(nx), int(ny)),
                "type": "semantic",
                "score": float(grid[self.map.PERSON_PROB, ny, nx]),
            })

        # 3. 재관측 후보 — 관측된 지 오래된 곳
        old_obs = np.argwhere((grid[self.map.OCCUPANCY] > 0) & (grid[self.map.LAST_OBS] > 30))
        for ny, nx in old_obs:
            candidates.append({
                "cell": (int(nx), int(ny)),
                "type": "reobserve",
                "score": min(float(grid[self.map.LAST_OBS, ny, nx]) / 100.0, 1.0),
            })

        return candidates

    def select_next(self, candidates, drone_x, drone_y):
        """규칙 기반 최고 점수 후보 선택 (거리 페널티 포함).
        → 향후 여기를 학습된 평가 모델 추론으로 교체 가능."""
        if not candidates:
            return None
        weights = {"semantic": 3.0, "frontier": 1.5, "reobserve": 1.0}
        cx, cy = self.map.local_to_cell(drone_x, drone_y)

        best, best_score = None, -1e9
        for c in candidates:
            nx, ny = c["cell"]
            dist_cells = ((nx - cx) ** 2 + (ny - cy) ** 2) ** 0.5
            score = c["score"] * weights.get(c["type"], 1.0) - dist_cells * 0.05
            if score > best_score:
                best_score = score
                best = c
        return best

    def next_waypoint(self, drone_x, drone_y):
        """다음 목표를 로컬 NED (x, y, z_ned, 후보유형) 로 반환. 후보가 없으면 None."""
        candidates = self.generate_candidates(drone_x, drone_y)
        best = self.select_next(candidates, drone_x, drone_y)
        if best is None:
            return None
        x, y = self.cell_to_local(*best["cell"])
        return x, y, -self.cruise_alt, best["type"]
