"""
bench_models.py — 시연 장비에서 모델별 추론 속도·VRAM 측정

왜 필요한가
  모델 선택을 GFLOPs 로만 정하면 틀린다. 실제 처리량은 GPU 아키텍처·메모리
  대역폭·후처리(NMS)에 좌우된다. **시연할 바로 그 장비에서 재야** 의미가 있다.

  시연은 RTX 4080 SUPER(16GB)에서 한다. 개발 노트북(4GB)에서 잰 값은
  그 장비를 대변하지 못하므로, 이 스크립트를 시연 장비에서 돌려야 한다.

가중치를 받지 않는 이유
  구조만 같으면 추론 시간과 VRAM 은 동일하다. 가중치 값은 속도에 영향이 없다.
  그래서 .yaml 로 무작위 초기화 모델을 만들어 잰다 — 다운로드 없이 즉시 측정된다.

시뮬레이터 동시 구동 주의
  언리얼을 같은 GPU 에서 돌리면 여기 나온 FPS 를 다 쓸 수 없다.
  '여유' 열을 보고 언리얼 몫을 남겨야 한다.

실행:
  python bench_models.py                          # 기본 후보 · imgsz 960/1280
  python bench_models.py --models yolo11s,yolo11m --imgsz 1280 --runs 50
"""
import argparse
import io
import contextlib
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

DEFAULT_MODELS = "yolov8s,yolo11s,yolo11m,yolo11l"
TARGET_FPS = 5.0          # 기준서 요건: 초당 5장 이상


def bench(name, imgsz, runs, warmup, device):
    from ultralytics import YOLO
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = YOLO(f"{name}.yaml", task="detect")
        model.model.to(device).eval()
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for _ in range(warmup):
            model(img, imgsz=imgsz, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(img, imgsz=imgsz, verbose=False, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return params, runs / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--imgsz", default="960,1280",
                    help="쉼표로 여러 개. 원본이 1280x720 이라 1280 이 원본 해상도다")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"장비: {p.name} · VRAM {p.total_memory / 2**30:.1f} GB")
    else:
        print("장비: CPU (GPU 미검출) — 이 수치는 시연 판단에 쓸 수 없다")
    print(f"PyTorch {torch.__version__}\n")

    sizes = [int(s) for s in args.imgsz.split(",")]
    print(f"{'모델':<10} {'imgsz':>6} {'파라미터':>9} {'FPS':>8} {'여유':>7} {'VRAM':>8}")
    print("-" * 54)
    for name in args.models.split(","):
        for imgsz in sizes:
            try:
                params, fps, peak = bench(name.strip(), imgsz, args.runs, args.warmup, device)
                margin = f"{fps / TARGET_FPS:.0f}배" if fps > 0 else "-"
                vram = f"{peak:.2f} GB" if peak == peak else "-"
                print(f"{name:<10} {imgsz:>6} {params:>8.2f}M {fps:>8.1f} {margin:>7} {vram:>8}")
            except torch.cuda.OutOfMemoryError:
                print(f"{name:<10} {imgsz:>6} {'':>9} {'VRAM 부족':>8}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{name:<10} {imgsz:>6} 오류: {str(e)[:40]}")
    print(f"\n'여유' = 기준서 요건({TARGET_FPS} FPS) 대비 배수.")
    print("언리얼을 같은 GPU 에서 돌리면 이 값을 다 쓸 수 없다 — 넉넉히 남길 것.")
    print("VRAM 은 추론 기준이다. 학습은 이보다 3~5배 더 든다.")


if __name__ == "__main__":
    main()
