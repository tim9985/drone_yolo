"""
train_status.py — 학습 진행 상황 한눈에 확인

실행: python train_status.py
      python train_status.py --stage 2      # stage2 학습 확인
      python train_status.py --rows 15      # 최근 15 epoch 표시
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent


def proc_alive():
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*train_person*' } | "
             "Select-Object -First 1).ProcessId"],
            capture_output=True, text=True, timeout=30)
        pid = r.stdout.strip()
        return pid if pid else None
    except Exception:
        return None


def gpu_line():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:
        return "조회 불가"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--name", default=None,
                    help="결과 폴더명 직접 지정. 생략하면 runs_person 에서 가장 최근 것을 자동 선택")
    ap.add_argument("--watch", type=int, nargs="?", const=30, default=0, metavar="SEC",
                    help="지정 초마다 화면을 새로 그린다 (기본 30초). Ctrl+C 로 종료")
    args = ap.parse_args()

    if args.watch:
        import os
        import time
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"[{time.strftime('%H:%M:%S')}] {args.watch}초마다 갱신 · Ctrl+C 로 종료")
                print()
                try:
                    _report(args)
                except Exception as e:
                    # 학습이 results.csv 를 쓰는 순간과 겹치면 읽기가 실패할 수 있다.
                    # 감시 도구가 죽으면 안 되므로 삼키고 다음 주기에 다시 읽는다.
                    print(f"조회 실패(무시): {e}")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print()
            print("감시 종료")
            return
    _report(args)


def _report(args):
    runs = BASE_DIR / "runs_person"
    if args.name:
        name = args.name
    else:
        # 가장 최근에 갱신된 실행을 자동 선택 (--name 으로 돌린 학습도 잡히도록)
        cands = [d for d in runs.glob("*") if (d / "results.csv").exists()] if runs.exists() else []
        if cands:
            name = max(cands, key=lambda d: (d / "results.csv").stat().st_mtime).name
        else:
            name = "stage1_nomad" if args.stage == 1 else "stage2_mixed"
    run_dir = BASE_DIR / "runs_person" / name
    csv_path = run_dir / "results.csv"

    pid = proc_alive()
    print(f"학습 프로세스 : {'실행 중 (PID ' + pid + ')' if pid else '실행 안 함'}")
    print(f"GPU           : {gpu_line()}")

    if not csv_path.exists():
        print(f"\n아직 결과 없음: {csv_path}")
        return

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        print("\n첫 epoch 진행 중 (아직 기록 없음)")
        return

    # 총 epoch 는 실행 설정(args.yaml)에서 읽는다.
    # stage 로 추정하면 --epochs 로 바꿔 돌린 실행에서 잔여 시간이 틀리게 나온다.
    total = 60 if args.stage == 1 else 20
    try:
        import yaml
        total = int(yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))["epochs"])
    except Exception:
        pass

    # 분류 학습은 지표 이름이 다르다(accuracy_top1). 탐지용 표를 그리면 KeyError 가 난다.
    if rows and "metrics/accuracy_top1" in rows[0]:
        print(f"\n=== {name}  {len(rows)}/{total} epoch  (분류) ===")
        print(f"{'ep':>3} {'top1':>8} {'top5':>8} {'train_loss':>11} {'val_loss':>10}")
        for x in rows[-args.rows:]:
            print(f"{int(float(x['epoch'])):>3} {float(x['metrics/accuracy_top1']):>8.4f} "
                  f"{float(x.get('metrics/accuracy_top5', 0)):>8.4f} "
                  f"{float(x.get('train/loss', 0)):>11.4f} {float(x.get('val/loss', 0)):>10.4f}")
        acc = [float(x["metrics/accuracy_top1"]) for x in rows]
        bi = acc.index(max(acc))
        print(f"\n최고 top1 {max(acc):.4f} (epoch {bi+1}) → best.pt 에 저장됨")
        print(f"미갱신 {len(acc)-1-bi} epoch")
        recent = rows[-min(len(rows), 5):]
        if len(recent) >= 2:
            per = (float(recent[-1]["time"]) - float(recent[0]["time"])) / (len(recent) - 1)
            if per > 0:
                left = (total - len(rows)) * per
                print(f"\nepoch당 {per:.0f}초 | 잔여 {total-len(rows)} epoch ≈ {left/60:.0f}분")
        print("\n※ 불균형(검증 person 82.6% : fallen 17.4%)이라 top1 만 보면 안 된다.")
        print("   쓰러짐 재현율은  python eval_pose_cls.py  로 따로 잰다.")
        return
    print(f"\n=== {name}  {len(rows)}/{total} epoch ===")
    print(f"{'ep':>3} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'box_loss':>9}")
    for r in rows[-args.rows:]:
        print(f"{int(float(r['epoch'])):>3} {float(r['metrics/mAP50(B)']):>7.3f} "
              f"{float(r['metrics/mAP50-95(B)']):>9.3f} "
              f"{float(r['metrics/precision(B)']):>7.3f} "
              f"{float(r['metrics/recall(B)']):>7.3f} "
              f"{float(r['train/box_loss']):>9.3f}")

    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    print(f"\n최고 mAP50 {float(best['metrics/mAP50(B)']):.3f} "
          f"(epoch {int(float(best['epoch']))}) → best.pt 에 저장됨")

    # 이어 학습 시 time 이 리셋되므로 최근 구간으로 epoch 시간 추정
    recent = rows[-min(len(rows), 5):]
    if len(recent) >= 2:
        per = (float(recent[-1]["time"]) - float(recent[0]["time"])) / (len(recent) - 1)
        left = (total - len(rows)) * per
        if per > 0:
            print(f"epoch당 {per:.0f}초 | 잔여 {total-len(rows)} epoch ≈ {left/3600:.1f}시간")

    # 완료 판정: 최종 가중치가 복사되어 있고 프로세스가 없으면 정상 종료.
    # patience 조기 종료가 걸리면 epoch 수가 total 보다 적어도 '완료'다.
    final_w = BASE_DIR / "weights" / f"yolov8s_{name}.pt"
    if pid is None:
        if final_w.exists() and final_w.stat().st_mtime >= csv_path.stat().st_mtime - 300:
            reason = "조기 종료(patience)" if len(rows) < total else "전체 epoch 소화"
            print(f"\n✔ 학습 완료 — {reason}. 최종 가중치: {final_w.name}")
        else:
            print("\n※ 프로세스가 없는데 완료 흔적이 없습니다. 이어서 하려면:")
            print(f"   python train_person.py --stage {args.stage} --resume")


if __name__ == "__main__":
    main()
