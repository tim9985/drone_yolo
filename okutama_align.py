"""
okutama_align.py — 라벨-영상 프레임 오프셋 추정

문제
  Okutama 배포 영상 10개 중 4개는 라벨의 최대 프레임 번호가 실제 디코딩 가능한
  프레임 수보다 크다. ffprobe 실측:
      1.2.10  실제 1651 / 메타 1671 / 라벨 최대 1690
      2.2.10  실제 1799 / 메타 1817 / 라벨 최대 1834
  그래서 라벨을 그대로 갖다 붙이면 상자가 사람 옆 빈 땅에 앉는다.

  다만 검수해 보니 **모든 프레임이 어긋나는 게 아니다** — 드론이 멈춰 있는
  구간은 맞고 움직이는 구간만 틀린다. 이는 프레임이 무작위로 빠진 게 아니라
  **일정한 오프셋만큼 밀렸을 때** 나타나는 양상이다. 오프셋을 찾으면 복구된다.

  이게 중요한 이유: 이 두 영상에 Lying 이 1,640 + 1,739 = 3,379 개 들어 있다.
  정상 6개 영상을 다 합쳐도 211 개뿐이라, 이 둘을 살리느냐가 자세 학습의
  성패를 가른다.

방법
  우리 사람 탐지기(yolov8s_stage1_all, 여름 mAP50 0.641)로 프레임에서 사람을
  찾은 뒤, 라벨을 여러 오프셋으로 밀어 보며 **가장 잘 겹치는 오프셋**을 고른다.
  탐지기가 자세는 못 가려도 사람 위치는 잘 잡으므로 정렬 기준으로 쓸 수 있다.

실행:
  python okutama_align.py                       # 손상 4개 전부
  python okutama_align.py --videos 1.2.10 2.2.10 --range 60
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "data" / "raw" / "okutama"
LAB_DIR = SRC / "Labels" / "MultiActionLabels" / "3840x2160"
WEIGHTS = BASE_DIR / "weights" / "yolov8s_stage1_all.pt"
POSE_SET = {"Standing", "Sitting", "Lying", "Walking", "Running"}
SUSPECT = ["1.2.10", "2.2.10", "1.2.3", "2.1.8"]


def load_labels(vid):
    """프레임 → [(x1,y1,x2,y2)] (4K 좌표). 자세 라벨 유무와 무관하게 사람 전부."""
    per = defaultdict(list)
    path = LAB_DIR / f"{vid}.txt"
    for ln in open(path, encoding="utf-8", errors="ignore"):
        q = ln.split()
        if len(q) < 10:
            continue
        try:
            x1, y1, x2, y2, f, lost = (int(q[1]), int(q[2]), int(q[3]),
                                       int(q[4]), int(q[5]), int(q[6]))
        except ValueError:
            continue
        if lost or x2 <= x1 or y2 <= y1:
            continue
        per[f].append((x1, y1, x2, y2))
    return per


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def score(dets, labs):
    """탐지 박스와 라벨 박스의 평균 최대 IoU. 둘 중 하나라도 비면 0."""
    if not dets or not labs:
        return 0.0
    return float(np.mean([max(iou(d, l) for l in labs) for d in dets]))


def main():
    ap = argparse.ArgumentParser(description="라벨-영상 프레임 오프셋 추정")
    ap.add_argument("--videos", nargs="*", default=SUSPECT)
    ap.add_argument("--range", type=int, default=60, help="탐색할 오프셋 범위 (+-)")
    ap.add_argument("--samples", type=int, default=12, help="검사할 프레임 수")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="정렬 기준이라 확실한 탐지만 쓴다 — 운용값 0.15보다 높게")
    ap.add_argument("--save", default=str(BASE_DIR / "configs" / "okutama_offsets.json"),
                    help="추정 결과를 저장할 경로. okutama_prep.py 가 이 파일을 읽는다")
    args = ap.parse_args()

    from ultralytics import YOLO
    import json
    model = YOLO(str(WEIGHTS))
    results = {}

    for vid in args.videos:
        mp4 = next(SRC.glob(f"Drone*/*/{vid}.mp4"), None)
        if mp4 is None or not (LAB_DIR / f"{vid}.txt").exists():
            print(f"{vid}: 파일 없음 — 건너뜀")
            continue
        per = load_labels(vid)
        if not per:
            print(f"{vid}: 라벨 없음")
            continue

        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 라벨이 조밀한 구간에서 고르게 뽑는다. 양 끝은 피한다(오프셋 탐색 여유 확보)
        lo, hi = args.range + 5, total - args.range - 5
        cand = [f for f in sorted(per) if lo <= f <= hi and len(per[f]) >= 1]
        if len(cand) < args.samples:
            print(f"{vid}: 표본 부족 ({len(cand)})")
            cap.release()
            continue
        step = len(cand) // args.samples
        picks = cand[::step][:args.samples]

        # 순차 디코딩하며 표본 위치의 프레임만 탐지한다
        dets_at = {}
        want = set(picks)
        fno = -1
        while fno < max(picks):
            ok, frame = cap.read()
            if not ok:
                break
            fno += 1
            if fno not in want:
                continue
            r = model(frame, verbose=False, imgsz=1280, conf=args.conf)[0]
            dets_at[fno] = [tuple(map(int, b)) for b in r.boxes.xyxy.cpu().numpy()]
        cap.release()

        # 오프셋별 점수. 디코딩 위치 p 의 화면이 라벨 프레임 p+d 에 해당하는지 본다
        best, curve = None, []
        for d in range(-args.range, args.range + 1):
            s = np.mean([score(dets_at.get(p, []), per.get(p + d, [])) for p in picks])
            curve.append((d, s))
            if best is None or s > best[1]:
                best = (d, s)
        zero = dict(curve)[0]
        top5 = sorted(curve, key=lambda x: -x[1])[:5]
        print(f"\n{vid}  (영상 {total}프레임 · 라벨 최대 {max(per)} · 표본 {len(picks)})")
        print(f"  오프셋 0  (보정 없음) 점수 {zero:.3f}")
        print(f"  최적 오프셋 {best[0]:+d}  점수 {best[1]:.3f}")
        print(f"  상위 5: " + "  ".join(f"{d:+d}:{s:.3f}" for d, s in top5))
        # 판정. 점수가 낮으면 오프셋을 신뢰할 수 없으므로 쓰지 않는다
        if best[1] <= 0.30:
            verdict, use = "정렬 실패 — 단순 밀림이 아님. 쓰지 말 것", None
        elif best[1] > zero + 0.05:
            verdict, use = f"오프셋 {best[0]:+d} 로 복구", best[0]
        else:
            verdict, use = "보정 불필요 (이미 정렬됨)", 0
        print(f"  → {verdict}")
        results[vid] = {"offset": use, "score": round(best[1], 3),
                        "score_at_zero": round(zero, 3),
                        "video_frames": total, "label_max_frame": max(per)}

    if args.save and results:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        # 기존 결과와 합친다 — 일부 영상만 다시 재도 나머지가 남도록
        old = {}
        if out.exists():
            try:
                old = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                pass
        old.update(results)
        out.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n저장: {out}  ({len(old)}개 영상)")


if __name__ == "__main__":
    main()
