"""
train_person.py — 사람 탐지 모델 학습 (2단계)

  stage1 : VisDrone 가중치 → NOMAD 실데이터 파인튜닝
  stage2 : stage1 결과 → NOMAD + 합성데이터 혼합 재파인튜닝 (쓰러진 자세 보강)

핵심 설계
  · 검증셋은 **항상 NOMAD 실데이터만** 쓴다. 합성 데이터로 검증하면 마네킹 질감에
    과적합된 모델이 좋아 보이는 착시가 생긴다.
  · stage2 의 합성 비율은 13% 수준(합성 411 / 전체 3068)으로 제한한다. 합성이 많으면
    금속 마네킹 질감을 학습해 실제 성능이 떨어진다.
  · 학습 해상도 = 추론 해상도(운용 조건). 크롭이 1280x720, 사람 94px 기준이므로
    imgsz 960 학습 → 추론도 960 으로 맞춘다.
  · NOMAD 는 여름·정오·미국 시골로 조명/계절이 단일하다. 이를 보완하려 색상·밝기
    증강(hsv_*)을 기본값보다 강하게 준다.

실행:
  python train_person.py --stage 1
  python train_person.py --stage 2                # stage1 최종 가중치에서 이어서
  python train_person.py --stage 1 --epochs 40 --batch 4 --imgsz 960
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
NOMAD_YAML = BASE_DIR / "data" / "dataset_nomad" / "data.yaml"
SYNTH_IMG = BASE_DIR / "data" / "dataset_synth" / "yolo" / "images" / "train"
MIXED_YAML = BASE_DIR / "configs" / "data_stage2.yaml"
BASE_WEIGHTS = BASE_DIR / "weights" / "yolov8s_visdrone.pt"
RUNS = BASE_DIR / "runs_person"


def schedule_shutdown(sec):
    """학습이 정상 완료된 뒤에만 호출된다. 취소: 다른 터미널에서 shutdown /a"""
    import subprocess
    print(f"\n※ {sec}초 뒤 컴퓨터를 종료합니다. 취소하려면 다른 터미널에서:  shutdown /a")
    try:
        subprocess.run(["shutdown", "/s", "/t", str(sec),
                        "/c", "Claude Code: 학습 완료로 자동 종료"], check=True)
    except Exception as e:
        print(f"종료 예약 실패(무시): {e}")


def write_mixed_yaml(repeat=3):
    """stage2용 data.yaml — 실데이터 전체 train + 합성 train, 검증은 **실데이터만**.

    합성을 검증에 넣으면 "자기가 만든 그림체를 자기가 알아보는가"를 재게 되어
    수치가 부풀려진다. 실사에서 좋아져야 의미가 있다.

    repeat: 합성 데이터를 몇 배로 넣을지. 합성 411장은 실사 14,342장의 2.8%뿐이라
      그대로 넣으면 효과가 있어도 묻힌다. 3배(약 8%)면 드러날 만하면서도
      합성 편향이 실사를 덮지 않는다.
    """
    D = BASE_DIR / "data"
    real_train = [D / n / "images" / "train" for n in
                  ("dataset_nomad", "dataset_nomad_a11_20", "dataset_nomad_a21_30")]
    real_train.append(D / "dataset_wisard" / "images" / "train")
    real_val = [D / "dataset_nomad" / "images" / "val",
                D / "dataset_nomad_a11_20" / "images" / "val",
                D / "dataset_wisard" / "images" / "val"]

    if repeat <= 1:
        train_block = "train:\n" + "".join(
            f"  - {d.as_posix()}\n" for d in real_train + [SYNTH_IMG])
    else:
        # 반복 투입은 폴더 지정으로 표현할 수 없어 이미지 목록 파일을 만든다.
        # 실사 크롭은 .jpg, 합성은 .png 다. 확장자를 고정하면 조용히 0장이 섞인다.
        def imgs(d):
            return sorted(q for q in d.iterdir()
                          if q.suffix.lower() in (".jpg", ".jpeg", ".png"))

        lst = BASE_DIR / "configs" / "stage2_train_list.txt"
        paths = []
        for d in real_train:
            paths += [str(q) for q in imgs(d)]
        n_real = len(paths)
        synth = [str(q) for q in imgs(SYNTH_IMG)]
        if not synth:
            raise SystemExit(f"합성 이미지를 찾지 못했습니다: {SYNTH_IMG}")
        paths += synth * repeat
        lst.write_text("\n".join(paths), encoding="utf-8")
        print(f"  학습 구성: 실사 {n_real}장 + 합성 {len(synth)}장×{repeat} "
              f"= {len(paths)}장 (합성 비중 {len(synth)*repeat/len(paths)*100:.1f}%)")
        train_block = f"train: {lst.as_posix()}\n"

    MIXED_YAML.write_text(
        f"# stage2: 실데이터 + 합성 혼합 (합성 {repeat}배). val 은 실데이터만\n"
        + train_block
        + "val:\n" + "".join(f"  - {d.as_posix()}\n" for d in real_val)
        + "nc: 1\nnames: ['person']\n",
        encoding="utf-8")
    return MIXED_YAML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--synth-repeat", type=int, default=3,
                    help="stage2 에서 합성 데이터를 몇 배로 넣을지 (1이면 원본 그대로)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", default=-1,
                    help="-1 이면 VRAM에 맞춰 자동 (RTX 3050 4.3GB 권장)")
    ap.add_argument("--weights", default=None, help="시작 가중치 직접 지정")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default=0)
    ap.add_argument("--resume", action="store_true",
                    help="중단된 학습을 last.pt 에서 이어서 진행")
    ap.add_argument("--data", default=None,
                    help="data.yaml 직접 지정 (예: data_nomad20.yaml — 배우 20명 재학습)")
    ap.add_argument("--name", default=None, help="결과 폴더명 직접 지정")
    ap.add_argument("--shutdown", type=int, default=0, metavar="SEC",
                    help="학습 정상 완료 후 지정 초 뒤 컴퓨터 종료 (예: --shutdown 120). "
                         "취소는 다른 터미널에서 'shutdown /a'")
    args = ap.parse_args()

    from ultralytics import YOLO

    # ── 이어 학습 ──
    if args.resume:
        # --name 을 준 실행(예: stage1_nomad20)을 이어받을 수 있어야 한다
        name = args.name or ("stage1_nomad" if args.stage == 1 else "stage2_synth")
        last = RUNS / name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"이어받을 체크포인트가 없습니다: {last}")
        print(f"=== stage{args.stage} 이어 학습: {last} ===")
        YOLO(str(last)).train(resume=True)
        best = RUNS / name / "weights" / "best.pt"
        if best.exists():
            dest = BASE_DIR / "weights" / f"yolov8s_{name}.pt"
            shutil.copy(best, dest)
            print(f"\n최종 가중치 복사: {dest}")
        if args.shutdown:
            schedule_shutdown(args.shutdown)
        return

    if args.stage == 1:
        data = NOMAD_YAML
        weights = Path(args.weights) if args.weights else BASE_WEIGHTS
        epochs = args.epochs or 60
        name = "stage1_nomad"
        # 단일 도메인(여름·정오)이라 색상 증강을 기본보다 강하게
        extra = dict(hsv_h=0.02, hsv_s=0.8, hsv_v=0.5, degrees=10.0,
                     translate=0.15, scale=0.5, fliplr=0.5, mosaic=1.0)
    else:
        data = write_mixed_yaml(args.synth_repeat)
        default_w = RUNS / "stage1_all" / "weights" / "best.pt"
        weights = Path(args.weights) if args.weights else default_w
        if not weights.exists():
            raise SystemExit(f"stage1 가중치가 없습니다: {weights}\n먼저 --stage 1 을 실행하세요.")
        epochs = args.epochs or 8
        name = "stage2_synth"
        # 이어 학습이므로 증강을 약하게, 학습률도 낮게
        extra = dict(hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=8.0,
                     translate=0.1, scale=0.4, fliplr=0.5, mosaic=0.5,
                     lr0=0.002)

    if args.data:
        data = Path(args.data)
    if args.name:
        name = args.name

    if not Path(data).exists():
        raise SystemExit(f"데이터 설정 없음: {data}")
    if not weights.exists():
        raise SystemExit(f"가중치 없음: {weights}")

    print(f"=== stage{args.stage} 학습 ===")
    print(f"  시작 가중치 : {weights}")
    print(f"  데이터      : {data}")
    print(f"  imgsz {args.imgsz} / epochs {epochs} / batch {args.batch} / patience {args.patience}")

    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch if args.batch == -1 else int(args.batch),
        device=args.device,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        patience=args.patience,
        amp=True,             # 4.3GB VRAM 에서 필수
        cache=False,          # 1.2GB 데이터셋을 RAM 캐시하면 오히려 불안정
        workers=4,
        seed=42,
        val=True,
        plots=True,
        **extra,
    )

    best = RUNS / name / "weights" / "best.pt"
    if best.exists():
        dest = BASE_DIR / "weights" / f"yolov8s_{name}.pt"
        shutil.copy(best, dest)
        print(f"\n최종 가중치 복사: {dest}")
    print(f"결과: {RUNS / name}")
    if args.shutdown:
        schedule_shutdown(args.shutdown)


if __name__ == "__main__":
    main()
