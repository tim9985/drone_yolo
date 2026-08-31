"""
train_pose_cls.py — 자세 분류기 학습 (2단계 구조의 2단)

구조
  stage1_all 로 사람 탐지 (탐지기 무손상)
      → 상자를 잘라 128x128 패치
      → **이 분류기**: 정상 / 쓰러짐

왜 분류기를 따로 두나
  탐지기를 2클래스로 바꾸는 시도(pose2)는 실패했다. 기본 학습률 0.01 이
  통합 모델의 특징을 망가뜨려 박스 손실이 1.2 → 1.9 로 나빠졌고,
  쓰러짐 재현율은 0.153 에 그쳤다.
  분류기를 분리하면 **겨울 mAP50 0.922 를 낸 탐지기를 전혀 건드리지 않는다.**

클래스 불균형
  학습 person 5,366 : fallen 2,258 (2.4:1). 검증은 4.7:1 로 더 심하다.
  정확도(accuracy)만 보면 "전부 person" 이라 답해도 82.6% 가 나오므로
  **쓰러짐 재현율을 따로 본다.** 평가는 eval_pose_cls.py 가 담당한다.

실행:
  python train_pose_cls.py                       # 기본 30 epoch
  python train_pose_cls.py --epochs 50 --model yolov8s-cls.pt
진행 확인:
  python train_status.py --name pose_cls --watch
"""
import argparse
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "dataset_pose_cls"
RUNS = BASE_DIR / "runs_person"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-cls.pt",
                    help="분류 백본. 패치가 128px 로 작아 n 으로 충분하다")
    ap.add_argument("--name", default="pose_cls")
    ap.add_argument("--data", default=str(DATA),
                    help="크롭 데이터셋 경로. 실사만 / 실사+합성 혼합을 바꿔 끼운다")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise SystemExit(f"크롭 데이터가 없습니다: {data_dir}\n"
                         "먼저 make_pose_crops.py 를 실행하세요.")

    from ultralytics import YOLO

    if args.resume:
        last = RUNS / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어받을 체크포인트가 없습니다: {last}")
        print(f"=== 이어 학습: {last} ===")
        YOLO(str(last)).train(resume=True)
    else:
        print("=== 자세 분류기 학습 ===")
        print(f"  백본   : {args.model}")
        print(f"  데이터 : {data_dir}")
        print(f"  설정   : imgsz {args.imgsz} / epochs {args.epochs} / batch {args.batch}")
        model = YOLO(args.model)
        model.train(
            data=str(data_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(RUNS),
            name=args.name,
            exist_ok=True,
            patience=args.patience,
            workers=4,
            seed=42,
            plots=True,
            # 항공 시점이라 상하좌우 어느 방향으로도 누울 수 있다.
            # 좌우 반전과 회전을 넉넉히 줘 방향 편향을 없앤다.
            fliplr=0.5,
            flipud=0.5,
            degrees=180.0,
            # 계절·조명 차이를 흡수 (탐지기 학습에서 쓴 값과 같은 취지)
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
            erasing=0.2,
        )

    best = RUNS / args.name / "weights" / "best.pt"
    if best.exists():
        dest = BASE_DIR / "weights" / f"{args.name}.pt"
        shutil.copy(best, dest)
        print(f"\n최종 가중치 복사: {dest}")
    print(f"결과: {RUNS / args.name}")
    print("평가:  python eval_pose_cls.py")


if __name__ == "__main__":
    main()
