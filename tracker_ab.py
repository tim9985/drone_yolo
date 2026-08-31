"""
tracker_ab.py — 추적 백엔드 A/B 측정 (flow · ByteTrack · BoT-SORT)

왜 측정하는가
  발표자료에는 ByteTrack 이라 적혀 있으나, 현재 구현은 광학흐름 + 칼만이다.
  "발표자료에 적었으니 그걸 쓴다" 보다 "재보고 골랐다" 가 강하다.
  그리고 드론은 **카메라가 움직이므로**, 카메라 움직임 보정(GMC)이 있는
  BoT-SORT 가 ByteTrack 보다 나을 가능성이 있다. 가정하지 말고 잰다.

방법
  같은 실비행 시나리오(run_experiment.py --patrol)를 백엔드만 바꿔가며 반복한다.
  환경변수 DRONE_TRACKER 로 전환하므로 코드를 고치지 않는다.

측정 항목
  처리 FPS            비행 중 실제 루프 속도
  처리 프레임 수      같은 시간에 몇 장을 소화했나
  탐지 건수           프레임 단위 원시 탐지
  확인 대상 수        병합 후 사람 단위 (KML 기준)
  미확인 후보 수      3회 미만 관측 — 낮을수록 트랙이 안정적
  미관측 비율         탐색 커버리지

주의
  시뮬레이터 실행마다 비행 궤적이 미세하게 달라 결과에 편차가 있다.
  --repeat 로 반복 평균을 내는 편이 안전하다.

실행:
  python tracker_ab.py                          # 3종 1회씩
  python tracker_ab.py --repeat 2               # 3종 2회씩
  python tracker_ab.py --backends flow,botsort
출력: metrics/tracker_ab.csv
"""
import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
OUT_CSV = BASE_DIR / "metrics" / "tracker_ab.csv"
PY = sys.executable

PATTERNS = {
    "frames":    re.compile(r"\[완료\] 총 (\d+)프레임"),
    "unobs":     re.compile(r"최종 미관측 비율: (\d+)%"),
    "confirmed": re.compile(r"확인 대상 (\d+)명"),
    "candidate": re.compile(r"미확인 후보 (\d+)건"),
    "dets":      re.compile(r"탐지 (\d+)건에서 병합"),
    "elapsed":   re.compile(r"총 소요 ([\d.]+)s"),
}
FPS_RE = re.compile(r"FPS=([\d.]+)")


def run_once(backend, timeout=900):
    """한 번의 실비행. 환경변수로 백엔드를 지정한다."""
    env = dict(os.environ)
    env["DRONE_TRACKER"] = backend
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run([PY, "-u", "run_experiment.py", "--patrol"],
                          cwd=str(BASE_DIR), env=env, capture_output=True,
                          timeout=timeout)
    log = (proc.stdout or b"").decode("utf-8", "ignore") + \
          (proc.stderr or b"").decode("utf-8", "ignore")

    row = {"backend": backend, "wall_s": round(time.time() - t0, 1),
           "exit": proc.returncode}
    for k, pat in PATTERNS.items():
        m = pat.search(log)
        row[k] = float(m.group(1)) if m else None
    fps = [float(v) for v in FPS_RE.findall(log)]
    # 첫 프레임들은 워밍업이라 0에 가깝다 — 중앙값이 대표값으로 안전하다
    row["fps_median"] = round(st.median(fps), 2) if fps else None
    row["fps_p10"] = round(st.quantiles(fps, n=10)[0], 2) if len(fps) >= 10 else None
    row["waypoints_ok"] = len(re.findall(r"waypoint \d+ 도달", log))
    return row, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="flow,bytetrack,botsort")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logdir = BASE_DIR / "logs"
    logdir.mkdir(exist_ok=True)

    rows = []
    total = len(backends) * args.repeat
    n = 0
    for rep in range(args.repeat):
        for b in backends:
            n += 1
            print(f"\n[{n}/{total}] {b} 실행 중... (약 3분)")
            try:
                row, log = run_once(b)
            except subprocess.TimeoutExpired:
                print(f"  시간 초과 — 건너뜀")
                continue
            (logdir / f"ab_{b}_{rep}.log").write_text(log, encoding="utf-8")
            row["repeat"] = rep
            rows.append(row)
            print(f"  프레임 {row['frames']} | FPS(중앙) {row['fps_median']} | "
                  f"탐지 {row['dets']}건 → 확인 {row['confirmed']}명 / 후보 {row['candidate']}건 | "
                  f"미관측 {row['unobs']}% | waypoint {row['waypoints_ok']}/4")

    if not rows:
        raise SystemExit("측정 결과가 없습니다.")

    keys = ["backend", "repeat", "frames", "fps_median", "fps_p10", "dets",
            "confirmed", "candidate", "unobs", "waypoints_ok", "wall_s", "exit"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    print("\n=== 백엔드별 요약 ===")
    hdr = f"{'백엔드':<11}{'프레임':>7}{'FPS중앙':>9}{'탐지':>7}{'확인대상':>9}{'후보':>7}{'미관측':>8}"
    print(hdr); print("-" * len(hdr))
    for b in backends:
        sub = [r for r in rows if r["backend"] == b]
        if not sub:
            continue
        def avg(k):
            v = [r[k] for r in sub if r.get(k) is not None]
            return st.mean(v) if v else float("nan")
        print(f"{b:<11}{avg('frames'):>7.0f}{avg('fps_median'):>9.2f}{avg('dets'):>7.0f}"
              f"{avg('confirmed'):>9.1f}{avg('candidate'):>7.1f}{avg('unobs'):>7.0f}%")
    print(f"\n저장: {out}")
    print("해석 — 확인 대상이 실제 인원에 가까울수록, 후보(3회 미만)가 적을수록 트랙이 안정적이다.")


if __name__ == "__main__":
    main()
