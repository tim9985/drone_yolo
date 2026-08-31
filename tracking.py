"""
tracking.py — 저주기 탐지 + 고주기 추적 (작업 3)

매 프레임 YOLO를 돌리는 대신, 탐지된 객체는 저렴한 OpenCV 추적기로 위치를 갱신해
GPU 부하를 줄인다. 칼만 필터로 프레임 간 위치를 스무딩하고 짧은 가림을 보간한다.

추적기 선택 주의
  과제에서 지정한 cv2.TrackerKCF_create / TrackerCSRT_create 는 **OpenCV 5.0에서
  제거**되었고 이 환경에는 legacy 모듈도 없다(설치본: opencv-python 5.0.0.93).
  따라서 사용 가능한 것 중에서 CSRT → KCF → MIL 순으로 자동 선택한다.
  현재 환경에서는 MIL 이 선택된다. opencv-contrib-python 을 설치하면 코드 변경
  없이 CSRT/KCF 로 승격된다.

주요 클래스
  Track               : 개별 추적 객체 (track_id, class, bbox, confidence, 마지막 갱신 프레임)
  MultiObjectTracker  : 탐지 결과 등록 + 추적 갱신 + 수명 관리
"""
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np


# ── 추적기 백엔드 선택 ──────────────────────────────────────────────────
def _tracker_factory():
    """사용 가능한 추적기 생성 함수와 이름을 반환 (CSRT > KCF > MIL 우선순위)."""
    for name in ("TrackerCSRT", "TrackerKCF", "TrackerMIL"):
        # OpenCV 4.x: cv2.TrackerXXX_create / 5.x: cv2.TrackerXXX.create
        fn = getattr(cv2, f"{name}_create", None)
        if fn is None:
            cls = getattr(cv2, name, None)
            fn = getattr(cls, "create", None) if cls is not None else None
        if fn is not None:
            return fn, name
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            fn = getattr(legacy, f"{name}_create", None)
            if fn is not None:
                return fn, f"legacy.{name}"
    raise RuntimeError("사용 가능한 OpenCV 추적기가 없습니다")


TRACKER_CREATE, TRACKER_NAME = _tracker_factory()


def iou(a, b):
    """bbox (x, y, w, h) 두 개의 IoU."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _make_kalman(cx, cy):
    """등속도 모델 칼만 필터. 상태 [cx, cy, vx, vy], 관측 [cx, cy]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
    return kf


@dataclass
class Track:
    track_id: int
    cls_name: str
    bbox: tuple                 # (x, y, w, h) 픽셀
    confidence: float
    last_update_frame: int      # 추적/탐지로 마지막 갱신된 프레임
    last_detect_frame: int      # YOLO로 마지막 확인된 프레임
    misses: int = 0             # 연속 추적 실패 횟수 (가림 구간)
    hits: int = 1               # 탐지로 확인된 누적 횟수
    kf: object = field(default=None, repr=False)
    cv_tracker: object = field(default=None, repr=False)
    flow_pts: object = field(default=None, repr=False)   # 광학흐름 특징점

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def as_dict(self):
        return {"track_id": self.track_id, "class": self.cls_name,
                "bbox": [round(v, 1) for v in self.bbox],
                "confidence": round(self.confidence, 3),
                "last_update_frame": self.last_update_frame,
                "last_detect_frame": self.last_detect_frame,
                "misses": self.misses, "hits": self.hits}


class MultiObjectTracker:
    """탐지 저주기 + 추적 고주기 구조의 다중 객체 추적기.

    max_misses  : 이 횟수만큼 연속 실패하면 트랙 폐기 (짧은 가림은 칼만 예측으로 유지)
    iou_match   : 새 탐지와 기존 트랙을 같은 객체로 볼 IoU 임계
    """

    # 광학흐름 파라미터 (LK 피라미드)
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    FLOW_PTS_PER_TRACK = 10

    FB_ERR_MAX = 2.0   # forward-backward 왕복 오차 허용치(px)

    def __init__(self, max_misses=3, iou_match=0.3, smooth=True, backend="flow"):
        """backend
          "flow" (기본) : 희소 광학흐름(LK). 트랙 전체를 한 번의 LK 호출로 처리해
                          매우 저렴하다. 카메라가 움직이고 대상이 정지한 본 과제
                          상황에 특히 잘 맞는다.
          "cv"          : OpenCV 추적기(CSRT/KCF/MIL 중 가용한 것).
                          ※ 이 환경에서는 MIL 만 가용한데, MIL 은 트랙당 수십 ms가
                            들어 YOLO보다 비싸다 — 실측상 오히려 느려지므로 기본값 아님.
        """
        self.tracks = {}
        self._next_id = 1
        self.max_misses = max_misses
        self.iou_match = iou_match
        self.smooth = smooth
        self.use_flow = (backend == "flow")
        self.backend = "OpticalFlowLK" if self.use_flow else TRACKER_NAME
        self._prev_gray = None

    # ── 내부 ──
    @staticmethod
    def _gray(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def _seed_flow_points(self, gray, t):
        """박스 안에서 추적할 특징점을 뽑는다. 특징이 부족하면 격자점으로 대체."""
        x, y, w, h = [int(v) for v in t.bbox]
        H, W = gray.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            t.flow_pts = None
            return
        roi = gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=self.FLOW_PTS_PER_TRACK,
                                      qualityLevel=0.01, minDistance=3)
        if pts is None or len(pts) < 3:
            gx = np.linspace(x0 + 2, x1 - 3, 3)
            gy = np.linspace(y0 + 2, y1 - 3, 3)
            pts = np.array([[[a, b]] for b in gy for a in gx], np.float32)
        else:
            pts = pts.astype(np.float32)
            pts[:, 0, 0] += x0
            pts[:, 0, 1] += y0
        t.flow_pts = pts

    def _init_tracker_for(self, frame, t):
        if self.use_flow:
            self._seed_flow_points(self._gray(frame), t)
            return
        x, y, w, h = t.bbox
        try:
            t.cv_tracker = TRACKER_CREATE()
            t.cv_tracker.init(frame, (int(x), int(y), int(max(w, 1)), int(max(h, 1))))
        except Exception:
            t.cv_tracker = None       # 초기화 실패 시 칼만 예측만 사용

    def _new_track(self, frame, det, frame_idx):
        x, y, w, h = det["bbox"]
        t = Track(track_id=self._next_id, cls_name=det["class"], bbox=(x, y, w, h),
                  confidence=det.get("confidence", 0.0),
                  last_update_frame=frame_idx, last_detect_frame=frame_idx)
        t.kf = _make_kalman(x + w / 2.0, y + h / 2.0)
        self._init_tracker_for(frame, t)
        self.tracks[t.track_id] = t
        self._next_id += 1
        return t

    def _kf_correct(self, t, cx, cy):
        if t.kf is None:
            return cx, cy
        t.kf.predict()
        est = t.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        return float(est[0].item()), float(est[1].item())

    # ── 공개 API ──
    def update_with_detections(self, frame, detections, frame_idx):
        """YOLO 탐지 프레임: 기존 트랙과 IoU로 연결하고, 없으면 새 트랙 생성."""
        unmatched = list(range(len(detections)))
        # 기존 트랙 중 신뢰도 높은 것부터 탐욕적으로 매칭
        for t in sorted(self.tracks.values(), key=lambda x: -x.hits):
            best, best_iou = None, self.iou_match
            for di in unmatched:
                v = iou(t.bbox, detections[di]["bbox"])
                if v > best_iou:
                    best, best_iou = di, v
            if best is None:
                continue
            det = detections[best]
            unmatched.remove(best)
            x, y, w, h = det["bbox"]
            cx, cy = x + w / 2.0, y + h / 2.0
            if self.smooth:
                cx, cy = self._kf_correct(t, cx, cy)
            t.bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
            t.confidence = det.get("confidence", t.confidence)
            t.cls_name = det["class"]
            t.last_update_frame = frame_idx
            t.last_detect_frame = frame_idx
            t.misses = 0
            t.hits += 1
            # 탐지로 보정된 위치에서 추적기를 재초기화 (드리프트 제거)
            self._init_tracker_for(frame, t)

        for di in unmatched:
            self._new_track(frame, detections[di], frame_idx)

        if self.use_flow:
            self._prev_gray = self._gray(frame)
        self._retire(frame_idx)
        return self.get_tracks()

    def update_tracking_only(self, frame, frame_idx):
        """추적 전용 프레임: YOLO 없이 추적기+칼만으로 위치 갱신."""
        if self.use_flow:
            return self._update_flow(frame, frame_idx)
        for t in list(self.tracks.values()):
            ok, box = False, None
            if t.cv_tracker is not None:
                try:
                    ok, box = t.cv_tracker.update(frame)
                except Exception:
                    ok = False
            if ok and box is not None:
                x, y, w, h = box
                cx, cy = x + w / 2.0, y + h / 2.0
                if self.smooth:
                    cx, cy = self._kf_correct(t, cx, cy)
                t.bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
                t.misses = 0
            else:
                # 추적 실패(가림 등) → 칼만 예측으로 몇 프레임 버틴다
                t.misses += 1
                if t.kf is not None:
                    pred = t.kf.predict()
                    cx, cy = float(pred[0].item()), float(pred[1].item())
                    _, _, w, h = t.bbox
                    t.bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
            t.last_update_frame = frame_idx
        self._retire(frame_idx)
        return self.get_tracks()

    def _update_flow(self, frame, frame_idx):
        """모든 트랙의 특징점을 한 번의 LK 호출로 일괄 추적 → 트랙별 중앙 변위 적용.
        트랙 수가 늘어도 비용이 거의 늘지 않아 MIL 대비 훨씬 저렴하다."""
        gray = self._gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return self.get_tracks()

        order, pts = [], []
        for t in self.tracks.values():
            if t.flow_pts is None or len(t.flow_pts) == 0:
                continue
            order.append((t, len(pts), len(pts) + len(t.flow_pts)))
            pts.extend(t.flow_pts)

        if pts:
            p0 = np.array(pts, np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None,
                                                 **self.LK_PARAMS)
            # forward-backward 검증: 되돌려 추적했을 때 원래 위치로 돌아오지 않는 점은 버린다.
            # 지면처럼 특징이 빈약한 영역에서 LK가 '성공'했다고 보고하며 엉뚱한 곳을
            # 따라가는 것을 막는다(유령 박스의 주원인).
            p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None,
                                                   **self.LK_PARAMS)
            fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)
            st = (st.reshape(-1).astype(bool) & stb.reshape(-1).astype(bool)
                  & (fb < self.FB_ERR_MAX))
        else:
            p1, st = None, None

        for t, i0, i1 in order:
            good_old = np.array(t.flow_pts, np.float32).reshape(-1, 2)[st[i0:i1]]
            good_new = p1.reshape(-1, 2)[i0:i1][st[i0:i1]]
            if len(good_new) < 3:
                t.misses += 1
                if t.kf is not None:      # 짧은 가림은 칼만 예측으로 이어붙인다
                    pred = t.kf.predict()
                    cx, cy = float(pred[0].item()), float(pred[1].item())
                    _, _, w, h = t.bbox
                    t.bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
                t.last_update_frame = frame_idx
                continue
            d = np.median(good_new - good_old, axis=0)
            x, y, w, h = t.bbox
            cx, cy = x + w / 2.0 + float(d[0]), y + h / 2.0 + float(d[1])
            if self.smooth:
                cx, cy = self._kf_correct(t, cx, cy)
            t.bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
            t.flow_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
            t.misses = 0
            t.last_update_frame = frame_idx

        # 특징점이 없어 추적 대상에서 빠진 트랙도 수명은 진행시킨다
        tracked = {t.track_id for t, _, _ in order}
        for t in self.tracks.values():
            if t.track_id not in tracked:
                t.misses += 1
                t.last_update_frame = frame_idx

        self._prev_gray = gray
        self._retire(frame_idx)
        return self.get_tracks()

    def _retire(self, frame_idx):
        for tid, t in list(self.tracks.items()):
            if t.misses > self.max_misses:
                del self.tracks[tid]

    def get_tracks(self):
        return list(self.tracks.values())

    def summary(self):
        return {"backend": self.backend, "active": len(self.tracks),
                "next_id": self._next_id}


def boxes_from_yolo(result, names):
    """ultralytics Results → [{'class','confidence','bbox'(x,y,w,h)}] 변환."""
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        dets.append({"class": names.get(int(b.cls[0]), str(int(b.cls[0]))),
                     "confidence": float(b.conf[0]),
                     "bbox": (x1, y1, x2 - x1, y2 - y1)})
    return dets


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"OpenCV {cv2.__version__} / 선택된 추적기 백엔드: {TRACKER_NAME}")

    # 합성 영상으로 동작 확인: 흰 배경에 검은 사각형이 등속 이동
    mot = MultiObjectTracker()
    W, H = 640, 480
    ids_seen = []
    for f in range(30):
        img = np.full((H, W, 3), 255, np.uint8)
        x = 50 + f * 8
        cv2.rectangle(img, (x, 200), (x + 60, 260), (30, 30, 30), -1)
        if f % 10 == 0:   # 10프레임마다 '탐지'
            dets = [{"class": "person", "confidence": 0.9, "bbox": (x, 200, 60, 60)}]
            tr = mot.update_with_detections(img, dets, f)
        else:
            tr = mot.update_tracking_only(img, f)
        if tr:
            ids_seen.append(tr[0].track_id)
    print(f"30프레임 중 추적 유지 {len(ids_seen)}프레임, "
          f"고유 track_id {sorted(set(ids_seen))} (1개면 ID 유지 성공)")
    print("최종 상태:", mot.summary())
    for t in mot.get_tracks():
        print("  ", t.as_dict())


# ── ByteTrack / BoT-SORT 백엔드 어댑터 ────────────────────────────────
class UltralyticsTracker:
    """ultralytics 내장 추적기(ByteTrack · BoT-SORT)를 기존 Track 규약에 맞춘다.

    왜 어댑터인가
      map_manager · kml_export · debug_overlay 가 모두 Track 객체를 받는다.
      백엔드만 갈아끼우고 호출부는 건드리지 않기 위해 같은 규약을 지킨다.

    광학흐름 백엔드와의 차이
      ultralytics 추적기는 **매 프레임 탐지**를 전제한다. 탐지 저주기 + 추적 고주기
      구조가 성립하지 않으므로 step() 한 번에 탐지와 연결이 함께 일어난다.

    ByteTrack  낮은 신뢰도 탐지를 2차 매칭에서 회수한다. 쓰러진 사람은 신뢰도가
               낮게 나오므로(임계값을 0.15 로 내린 이유) 이 구조가 유리하다.
    BoT-SORT   여기에 카메라 움직임 보정(GMC)이 더해진다. 드론은 기체가 움직이므로
               프레임 간 상자 이동이 커서 보정이 필요할 수 있다. 측정으로 판단한다.
    """

    CONFIGS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}

    def __init__(self, model, backend="bytetrack", imgsz=960, conf=0.15, max_misses=3):
        if backend not in self.CONFIGS:
            raise ValueError(f"지원하지 않는 백엔드: {backend} (가능: {list(self.CONFIGS)})")
        self.model = model
        self.cfg = self.CONFIGS[backend]
        self.backend = backend
        self.imgsz = imgsz
        self.conf = conf
        self.max_misses = max_misses
        self.tracks = {}
        self.last_dets = []          # debug_overlay 용 — 이번 프레임 탐지 상자
        self.last_plot = None        # results/frame_*.png 저장용 주석 이미지

    def step(self, frame, frame_idx):
        """매 프레임 호출. 탐지 + 연결을 한 번에 하고 Track 목록을 돌려준다."""
        res = self.model.track(frame, persist=True, tracker=self.cfg,
                               imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        seen = set()
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            name = self.model.names.get(int(b.cls[0]), str(int(b.cls[0])))
            c = float(b.conf[0])
            dets.append({"class": name, "confidence": c, "bbox": bbox})
            if b.id is None:          # 추적기가 아직 ID를 안 붙인 탐지
                continue
            tid = int(b.id[0])
            t = self.tracks.get(tid)
            if t is None:
                self.tracks[tid] = Track(track_id=tid, cls_name=name, bbox=bbox,
                                         confidence=c, last_update_frame=frame_idx,
                                         last_detect_frame=frame_idx)
            else:
                t.bbox, t.confidence, t.cls_name = bbox, c, name
                t.last_update_frame = t.last_detect_frame = frame_idx
                t.hits += 1
                t.misses = 0
            seen.add(tid)
        self.last_dets = dets
        self.last_plot = res.plot()

        # 이번 프레임에 안 잡힌 트랙은 폐기 카운트를 올린다.
        # ultralytics 도 자체 수명 관리를 하지만, hits/misses 는 우리 신고 게이트와
        # 오버레이가 쓰는 값이라 규약을 맞춰 둔다.
        for tid, t in list(self.tracks.items()):
            if tid not in seen:
                t.misses += 1
                if t.misses > self.max_misses:
                    del self.tracks[tid]
        return list(self.tracks.values())

    def get_tracks(self):
        return list(self.tracks.values())
