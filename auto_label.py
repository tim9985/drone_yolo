"""
auto_label.py — seg ground-truth 에서 YOLO 라벨 자동 생성 (작업 3)

입력: dataset_synth/raw/  (rgb_XXXX.png + seg_XXXX.png, collect_data.py --synth 산출물)
      seg_color_map.json  (set_segmentation_ids.py 가 생성한 인스턴스별 RGB 색상)
처리: 인스턴스별 색상 마스크 → 모폴로지 정리 → connectedComponents → 외접 사각형 → YOLO txt
      인스턴스마다 색이 달라 개체가 겹쳐도 분리 라벨링됨
출력: dataset_synth/yolo/{images,labels}/{train,val} + data.yaml
      dataset_synth/qc/qc_XXXX.png  (라벨 시각화 검수 이미지)

실행: python auto_label.py [--raw dataset_synth/raw] [--min-area 50] [--qc 10]
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
COLOR_MAP_JSON = BASE_DIR / "configs" / "seg_color_map.json"

CLASSES = {0: "person", 1: "vehicle"}
VAL_RATIO = 0.2
SEED = 42


def imread_u(path):
    """cv2.imread는 Windows에서 한글 경로(캡스톤)를 열지 못한다 → numpy 경유."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def mask_for_color(seg, color, tol=2):
    """seg 영상에서 특정 색상(±tol)의 마스크 추출.
    seg 배열은 collect 시 raw 바이트(RGB) 그대로 imwrite 되었고 imread로 되읽으면
    동일 배열이 복원되므로, seg_color_map.json 의 rgb 값을 그대로 비교한다."""
    lower = np.clip(np.array(color, dtype=np.int16) - tol, 0, 255).astype(np.uint8)
    upper = np.clip(np.array(color, dtype=np.int16) + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(seg, lower, upper)


def boxes_from_mask(mask, min_area, min_side=10, merge_fragments=True):
    """한 인스턴스(=한 색) 마스크에서 외접 사각형 추출.

    인스턴스 segmentation이므로 같은 색은 반드시 같은 개체다. 따라서 잔해·구조물에
    가려 마스크가 여러 조각으로 갈라져도 **하나의 박스로 병합**해야 한다
    (merge_fragments=True). 조각별로 박스를 만들면 한 사람이 여러 개로 라벨링된다.

    min_side: 한 변이 이 픽셀보다 작은 박스는 버린다. 본 프로젝트의 탐지 하한이
    '대상 20px 이상'이므로 그보다 훨씬 작은 조각은 학습에 해롭다.
    """
    # 팔다리 등으로 마스크가 조각나는 것을 모폴로지 닫힘으로 우선 병합
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # 노이즈 조각(단독으로는 버릴 크기)도 병합 대상에는 포함시킨다.
    # 가려진 손·발 조각까지 박스 범위에 넣어야 실제 개체 범위가 나온다.
    NOISE_PX = 12
    kept = [stats[i] for i in range(1, n) if stats[i][4] >= NOISE_PX]
    if not kept:
        return []

    if merge_fragments:
        x0 = min(s[0] for s in kept)
        y0 = min(s[1] for s in kept)
        x1 = max(s[0] + s[2] for s in kept)
        y1 = max(s[1] + s[3] for s in kept)
        area = sum(s[4] for s in kept)
        w, h = x1 - x0, y1 - y0
        if area < min_area or w < min_side or h < min_side:
            return []
        return [(x0, y0, w, h)]

    boxes = []
    for x, y, w, h, area in kept:
        if area < min_area or w < min_side or h < min_side:
            continue
        boxes.append((x, y, w, h))
    return boxes


def to_yolo_line(cls_id, x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(BASE_DIR / "data" / "dataset_synth" / "raw"))
    parser.add_argument("--min-area", type=int, default=100, help="최소 픽셀 면적(노이즈 제외)")
    parser.add_argument("--min-side", type=int, default=10, help="박스 최소 변 길이(px)")
    parser.add_argument("--qc", type=int, default=10, help="검수 이미지 수")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_root = raw_dir.parent / "yolo"
    qc_dir = raw_dir.parent / "qc"

    with open(COLOR_MAP_JSON, encoding="utf-8") as f:
        cmap = json.load(f)
    person_colors = [e["rgb"] for e in cmap.get("person", [])]
    vehicle_colors = [e["rgb"] for e in cmap.get("vehicle", [])]
    if not person_colors:
        raise SystemExit("seg_color_map.json 에 person 색상 없음 — set_segmentation_ids.py 먼저 실행")
    print(f"사람 인스턴스 {len(person_colors)}색 / 차량 {len(vehicle_colors)}색으로 라벨링")

    rgb_files = sorted(raw_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise SystemExit(f"수집 이미지 없음: {raw_dir}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    stats_total = {"images": 0, "person_boxes": 0, "vehicle_boxes": 0, "empty_images": 0}
    qc_saved = 0

    for rgb_path in rgb_files:
        idx = rgb_path.stem.split("_")[1]
        seg_path = raw_dir / f"seg_{idx}.png"
        if not seg_path.exists():
            print(f"[skip] {rgb_path.name}: seg 짝 없음")
            continue

        rgb = imread_u(rgb_path)
        seg = imread_u(seg_path)
        img_h, img_w = seg.shape[:2]

        lines = []
        # 인스턴스별로 개별 마스크 → 개체가 붙어 있어도 각각의 박스가 나온다
        person_boxes = []
        for color in person_colors:
            person_boxes.extend(boxes_from_mask(mask_for_color(seg, color), args.min_area, args.min_side))
        for (x, y, w, h) in person_boxes:
            lines.append(to_yolo_line(0, x, y, w, h, img_w, img_h))
        vehicle_boxes = []
        for color in vehicle_colors:
            vehicle_boxes.extend(boxes_from_mask(mask_for_color(seg, color), args.min_area, args.min_side))
        for (x, y, w, h) in vehicle_boxes:
            lines.append(to_yolo_line(1, x, y, w, h, img_w, img_h))

        split = "val" if random.random() < VAL_RATIO else "train"
        shutil.copy(rgb_path, out_root / "images" / split / rgb_path.name)
        label_path = out_root / "labels" / split / f"{rgb_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats_total["images"] += 1
        stats_total["person_boxes"] += len(person_boxes)
        stats_total["vehicle_boxes"] += len(vehicle_boxes)
        if not lines:
            stats_total["empty_images"] += 1

        # 검수 이미지: 라벨을 rgb 위에 그림
        if qc_saved < args.qc:
            vis = rgb.copy()
            for (x, y, w, h) in person_boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis, "person", (x, max(y - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            for (x, y, w, h) in vehicle_boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(vis, "vehicle", (x, max(y - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            imwrite_u(qc_dir / f"qc_{idx}.png", vis)
            qc_saved += 1

    # data.yaml — 차량 인스턴스가 없으면 person 단일 클래스로 기록한다
    # (표본이 0인 클래스를 정의에 남기면 학습 설정이 혼란스러워짐)
    names = list(CLASSES.values()) if vehicle_colors else ["person"]
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )

    print(f"\n=== auto_label 완료 ===")
    print(f"이미지 {stats_total['images']}장 | person 박스 {stats_total['person_boxes']} | "
          f"vehicle 박스 {stats_total['vehicle_boxes']} | 라벨 없는 이미지 {stats_total['empty_images']}")
    print(f"YOLO 데이터셋: {out_root}")
    print(f"검수 이미지 {qc_saved}장: {qc_dir}")
    return stats_total


if __name__ == "__main__":
    main()
