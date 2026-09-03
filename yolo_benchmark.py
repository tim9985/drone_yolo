"""
yolo_benchmark.py — AirSim 영상에 YOLO 추론 + FPS 측정 (M6)
전제: collect_data.py 로 dataset/rgb_0000.png 생성됨 (없으면 아무 이미지나 지정 가능)
실행: python yolo_benchmark.py
      python yolo_benchmark.py --img dataset/rgb_0003.png --model yolov8s.pt
성공 기준: FPS가 목표(5Hz)를 여유롭게 초과 + 결과 이미지에 탐지 박스 표시
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="data/misc/depth_samples/rgb_0000.png")
    parser.add_argument("--model", default="yolov8n.pt", help="첫 실행 시 자동 다운로드")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"이미지를 찾을 수 없음: {args.img} — collect_data.py 먼저 실행")

    print(f"모델 로드: {args.model}")
    model = YOLO(args.model)

    # 워밍업
    for _ in range(5):
        model(img, verbose=False)

    # 벤치마크
    t0 = time.time()
    for _ in range(args.runs):
        results = model(img, verbose=False)
    elapsed = time.time() - t0
    fps = args.runs / elapsed

    # 마지막 결과 시각화 저장
    annotated = results[0].plot()
    cv2.imwrite("yolo_result.png", annotated)

    n_det = len(results[0].boxes)
    print(f"\n=== 벤치마크 결과 ===")
    print(f"이미지: {args.img} ({img.shape[1]}x{img.shape[0]})")
    print(f"추론 속도: {fps:.1f} FPS  (목표 5Hz 대비 {'통과 ✓' if fps >= 5 else '미달 ✗ — 해상도 축소/경량 모델 검토'})")
    print(f"탐지 객체 수: {n_det}개 → yolo_result.png 확인")
    print("참고: Blocks 맵은 큐브뿐이라 탐지 0개가 정상. FPS 수치가 M6의 핵심")


if __name__ == "__main__":
    main()
