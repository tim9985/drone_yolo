"""
debug_overlay.py — 탐지·추적·Depth·좌표변환 통합 확인용 스냅샷 (작업 4)

UI가 아니라 디버그용 PNG 저장이다. 한 프레임에 네 가지가 동시에 맞게 작동하는지
한 장으로 판단하기 위한 것.

  1. YOLO 탐지 박스 (클래스명, 신뢰도) — 노란색
  2. 추적 중인 객체 — 청록색 + track_id
  3. 좌상단 텍스트: 계산 좌표 vs ground truth, 오차(m)
  4. 우측 하단 Depth 컬러맵 썸네일 (depth_utils 재사용)

사용:
  from debug_overlay import save_debug_frame
  save_debug_frame(rgb, frame_idx, detections=..., tracks=..., depth=..., info_lines=[...])
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

from depth_utils import depth_thumbnail, imwrite_u

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results"

COLOR_DET = (0, 220, 255)     # 탐지: 노랑
COLOR_TRK = (255, 220, 0)     # 추적: 청록
COLOR_TXT = (255, 255, 255)


def _text(img, s, org, scale=0.55, color=COLOR_TXT, thick=1):
    # 외곽선이 두꺼우면 글자가 겹쳐 보여 오히려 읽기 어렵다 → +2 로 제한
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_detections(img, detections):
    for d in detections or []:
        x, y, w, h = [int(v) for v in d["bbox"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_DET, 2)
        _text(img, f"{d['class']} {d.get('confidence', 0):.2f}",
              (x, max(y - 6, 14)), 0.5, COLOR_DET)
    return img


def draw_tracks(img, tracks):
    for t in tracks or []:
        x, y, w, h = [int(v) for v in t.bbox]
        # 가림 구간(칼만 예측 중)은 점선처럼 얇게 그려 구분
        thick = 2 if t.misses == 0 else 1
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_TRK, thick)
        tag = f"ID{t.track_id}"
        if t.misses:
            tag += f" (predict {t.misses})"
        _text(img, tag, (x, y + h + 16), 0.5, COLOR_TRK)
    return img


def draw_depth_thumb(img, depth, size=(200, 150), margin=10):
    if depth is None:
        return img
    th = depth_thumbnail(depth, size=size)
    H, W = img.shape[:2]
    x0, y0 = W - size[0] - margin, H - size[1] - margin
    if x0 < 0 or y0 < 0:
        return img
    img[y0:y0 + size[1], x0:x0 + size[0]] = th
    cv2.rectangle(img, (x0, y0), (x0 + size[0], y0 + size[1]), (255, 255, 255), 1)
    _text(img, "depth", (x0 + 4, y0 - 6), 0.45)
    return img


def save_debug_frame(rgb, frame_idx, detections=None, tracks=None, depth=None,
                     info_lines=None, out_dir=OUT_DIR, prefix="debug_frame"):
    """한 프레임에 네 요소를 모두 그려 저장하고 경로를 반환."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = rgb.copy()

    draw_detections(img, detections)
    draw_tracks(img, tracks)
    draw_depth_thumb(img, depth)

    y = 24
    _text(img, f"frame {frame_idx:04d}  det={len(detections or [])}  trk={len(tracks or [])}",
          (10, y), 0.6)
    for line in (info_lines or []):
        y += 22
        _text(img, line, (10, y), 0.55)

    path = out_dir / f"{prefix}_{frame_idx:04d}.png"
    imwrite_u(path, img)
    return path


def format_coord_line(est_ned=None, gt_ned=None, gps=None):
    """좌상단에 넣을 좌표/오차 한 줄 생성."""
    parts = []
    if est_ned is not None:
        parts.append(f"est NED ({est_ned[0]:.1f}, {est_ned[1]:.1f})")
    if gps is not None:
        parts.append(f"GPS ({gps[0]:.6f}, {gps[1]:.6f})")
    if est_ned is not None and gt_ned is not None:
        err = ((est_ned[0] - gt_ned[0]) ** 2 + (est_ned[1] - gt_ned[1]) ** 2) ** 0.5
        parts.append(f"GT ({gt_ned[0]:.1f}, {gt_ned[1]:.1f})  err {err:.2f}m")
    return " | ".join(parts)


if __name__ == "__main__":
    # 합성 프레임으로 레이아웃 확인
    from tracking import MultiObjectTracker

    W, H = 1280, 720
    rgb = np.full((H, W, 3), 120, np.uint8)
    cv2.rectangle(rgb, (500, 300), (620, 400), (60, 60, 60), -1)
    depth = np.full((H, W), 25.0, np.float32)
    depth[300:400, 500:620] = 18.0

    dets = [{"class": "person", "confidence": 0.87, "bbox": (500, 300, 120, 100)}]
    mot = MultiObjectTracker()
    tracks = mot.update_with_detections(rgb, dets, 0)

    line = format_coord_line(est_ned=(12.3, -4.5), gt_ned=(12.0, -4.2),
                             gps=(35.179612, 129.075548))
    p = save_debug_frame(rgb, 1, detections=dets, tracks=tracks, depth=depth,
                          info_lines=[line, f"tracker backend: {mot.backend}"])
    print("저장:", p)
