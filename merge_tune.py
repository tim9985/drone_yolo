"""
merge_tune.py — 탐지 병합 반경·최소 관측횟수 조정

왜 필요한가
  KML 은 원본 탐지 수백 건을 "대상 N명" 으로 병합해 내보낸다. 이때
  병합 반경이 좌표 오차보다 작으면 **한 사람이 여러 명으로 쪼개진다.**
  고도 16m 재정립 후 실제로 그랬다 — 실제 15명이 27명으로 신고됐다.

  좌표 오차는 고도에 비례해 커지므로 반경은 고도마다 다시 골라야 한다.
  비행을 반복하지 않고 정하기 위해, patrol_detect 가 남긴
  results/detections_raw.csv (병합 전 원본) 를 읽어 값만 바꿔 재분석한다.

정답 기준
  configs/seg_color_map.json 의 인스턴스별 ned 좌표를 그라운드 트루스로 쓴다.
  언리얼을 띄우지 않아도 되고, 합성 데이터셋 생성 때 이미 검증된 값이다.

실행:
  python merge_tune.py                          # 반경·게이트 격자 탐색
  python merge_tune.py --csv other_run.csv --tol 3.0
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "results" / "detections_raw.csv"
GT_JSON = BASE_DIR / "configs" / "seg_color_map.json"


def load_gt(path=GT_JSON):
    """배치된 마네킹의 홈 기준 NED 좌표 (보정용 표본도 사람이므로 포함한다)"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(e["name"], e["ned"][0], e["ned"][1]) for e in d["person"]]


def load_raw(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), r["class"], float(r["conf"]),
                         float(r["north_m"]), float(r["east_m"]),
                         r["track_id"] or None,
                         float(r["roll_deg"]) if r.get("roll_deg") else None,
                         float(r["pitch_deg"]) if r.get("pitch_deg") else None))
    return rows


def merge(raw, radius, min_hits):
    """kml_export 와 같은 규칙: track_id 우선 → 공간 반경 → 중앙값 위치"""
    targets = []          # {ids, pts, confs}
    by_id = {}
    for t, cls, conf, n, e, tid, *_ in raw:
        tgt = by_id.get(tid) if tid else None
        if tgt is None:
            best, bd = None, 1e9
            for g in targets:
                px = sorted(p[0] for p in g["pts"])[len(g["pts"]) // 2]
                py = sorted(p[1] for p in g["pts"])[len(g["pts"]) // 2]
                d = math.hypot(px - n, py - e)
                if d < bd:
                    best, bd = g, d
            tgt = best if best is not None and bd <= radius else None
        if tgt is None:
            tgt = {"pts": [], "confs": []}
            targets.append(tgt)
        tgt["pts"].append((n, e))
        tgt["confs"].append(conf)
        if tid:
            by_id[tid] = tgt
    out = []
    for g in targets:
        if len(g["pts"]) < min_hits:
            continue
        xs = sorted(p[0] for p in g["pts"]); ys = sorted(p[1] for p in g["pts"])
        out.append((xs[len(xs) // 2], ys[len(ys) // 2], len(g["pts"])))
    return out


def score(reported, gt, tol):
    """관측 많은 대상부터 그리디 매칭 (같은 사람에 둘이 붙는 것을 막는다)"""
    used, tp, errs = set(), 0, []
    for n, e, hits in sorted(reported, key=lambda r: -r[2]):
        best, bd = None, 1e9
        for i, (_, gx, gy) in enumerate(gt):
            if i in used:
                continue
            d = math.hypot(gx - n, gy - e)
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= tol:
            used.add(best); tp += 1; errs.append(bd)
    p = tp / len(reported) if reported else 0.0
    r = tp / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, (sum(errs) / len(errs) if errs else float("nan")), len(reported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--tol", type=float, default=4.0,
                    help="정답과 이 거리 안이면 맞춘 것으로 본다")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"원본 탐지 CSV 가 없습니다: {csv_path}\n"
                         "patrol_detect.py 를 한 번 실행하면 생성됩니다.")
    raw, gt = load_raw(csv_path), load_gt()
    print(f"원본 탐지 {len(raw)}건 · 정답 {len(gt)}명 · 매칭 허용 {args.tol}m\n")
    print(f"{'반경(m)':>7} {'게이트':>6} {'신고':>5} {'정밀도':>7} {'재현율':>7} {'F1':>7} {'오차(m)':>8}")
    best = None
    for radius in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        for mh in (3, 5, 8, 12):
            p, r, f1, err, n = score(merge(raw, radius, mh), gt, args.tol)
            print(f"{radius:>7.1f} {mh:>6d} {n:>5d} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {err:>8.2f}")
            if best is None or f1 > best[0]:
                best = (f1, radius, mh, p, r, n)
        print()
    f1, radius, mh, p, r, n = best
    print(f"권장: MERGE_RADIUS_M = {radius} · MIN_HITS = {mh}")
    print(f"      신고 {n}명 (정답 {len(gt)}명) · 정밀도 {p:.3f} · 재현율 {r:.3f} · F1 {f1:.3f}")


if __name__ == "__main__":
    main()
