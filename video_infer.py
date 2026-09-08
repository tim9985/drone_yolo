"""
video_infer.py — 항공 영상에 탐지+자세 모델을 걸어 실주행 검증

왜 필요한가
  지금까지의 수치는 전부 **정답이 붙은 정지 이미지** 기준이다. 실기체에 올리기
  전에 알아야 할 것이 두 가지 더 있다.

    (1) 정답이 없는 실제 연속 영상에서 눈으로 봐도 말이 되는가
    (2) 실시간이 되는가 — 프레임당 몇 ms 인가

  특히 (1)이 중요하다. 데이터셋 검증셋은 우리가 만든 크롭이라 구도가 균일하지만
  실제 비행 영상은 고도·각도·조명이 계속 변한다. 여기서 처음 드러나는 실패가 있다.

실시간 창을 띄우지 않는다
  결과는 **주석 영상 파일 + CSV + 콘솔 통계**로만 낸다. 화면 출력은 원격/무인
  실행에서 못 쓰고, 프로젝트 방침이기도 하다. 영상은 나중에 열어 보면 된다.

무엇을 재나
  · 프레임당 추론 시간 (전처리/추론/후처리 분리)
  · 클래스별 탐지 수와 신뢰도 분포
  · fallen 로 판정된 프레임 목록 — 나중에 눈으로 확인할 지점

주의
  이 스크립트는 **정답과 비교하지 않는다.** 정답이 없는 영상에 쓰는 도구다.
  정량 평가는 eval_pose3.py 를 쓸 것.

실행
  python video_infer.py --video data/raw/okutama/Drone2/Noon/2.2.10.mp4 \
      --weights weights/yolov8s_pose3_sn_freeze.pt --max-frames 900
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# BGR. fallen 은 눈에 띄어야 하므로 빨강, ambiguous 는 노랑(보류)
COLORS = {"person": (90, 190, 90), "fallen": (60, 60, 235), "ambiguous": (40, 190, 235)}
DEFAULT_COLOR = (200, 200, 200)


def imwrite_u(path, img):
    """한글 경로에서 cv2.imwrite 는 실패한다."""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def draw(frame, boxes, names, show_conf=True):
    for (x1, y1, x2, y2), cls, cf in boxes:
        nm = names.get(cls, str(cls))
        col = COLORS.get(nm, DEFAULT_COLOR)
        # fallen 은 굵게 — 운용자가 놓치면 안 되는 대상
        th = 3 if nm == "fallen" else 2
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
        tag = f"{nm} {cf:.2f}" if show_conf else nm
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1) - tht - 6),
                      (int(x1) + tw + 4, int(y1)), col, -1)
        cv2.putText(frame, tag, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description="항공 영상 탐지+자세 실주행 검증")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="weights/yolov8s_pose3_sn_freeze.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="영상은 정답이 없어 오탐이 눈에 거슬린다. 정지 이미지 평가(0.15)보다 높게 잡는다")
    ap.add_argument("--max-frames", type=int, default=0, help="0 이면 전체")
    ap.add_argument("--stride", type=int, default=1, help="N 프레임마다 1장 추론")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-video", action="store_true", help="주석 영상 저장 생략 (측정만)")
    ap.add_argument("--snapshot-fallen", type=int, default=12,
                    help="fallen 로 판정된 프레임을 최대 N 장 따로 저장")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        raise SystemExit(f"영상이 없습니다: {vp}")

    out_dir = Path(args.out_dir) if args.out_dir else (BASE_DIR / "runs_video" / vp.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "fallen_snapshots"

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"=== 영상 ===")
    print(f"  파일   : {vp.name}")
    print(f"  해상도 : {W}x{H} · {fps:.1f} fps · 총 {n_total:,} 프레임")
    print(f"  모델   : {Path(args.weights).name} · 클래스 {list(names.values())}")
    print(f"  설정   : imgsz {args.imgsz} · conf {args.conf} · stride {args.stride}\n")

    writer = None
    if not args.no_video:
        # 원본 해상도가 4K 면 파일이 너무 커진다. 긴 변 1280 으로 줄여 저장한다.
        scale = min(1.0, 1280 / max(W, H))
        ow, oh = int(W * scale), int(H * scale)
        vpath = out_dir / f"{vp.stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / args.stride, (ow, oh))
        if not writer.isOpened():
            print("  주석 영상 저장 불가 — 측정만 진행한다")
            writer = None

    rows = []
    cls_count = Counter()
    conf_sum = defaultdict(float)
    t_infer = []
    n_done = 0
    n_snap = 0
    fno = -1
    # 스냅샷 최소 간격 — 처리할 프레임 수를 예산으로 나눈다
    n_plan = args.max_frames or max(1, n_total // max(args.stride, 1))
    snap_gap = max(1, n_plan // max(args.snapshot_fallen, 1))
    last_snap_at = -snap_gap
    t0 = time.perf_counter()

    # 순차 디코딩만 쓴다. H.264 는 cap.set(POS_FRAMES) 로 건너뛰면 어긋난다
    # (Okutama 정렬 작업에서 실제로 겪은 문제).
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fno += 1
        if fno % args.stride:
            continue
        if args.max_frames and n_done >= args.max_frames:
            break

        ts = time.perf_counter()
        r = model(frame, verbose=False, imgsz=args.imgsz, conf=args.conf,
                  device=args.device)[0]
        t_infer.append((time.perf_counter() - ts) * 1000)

        boxes = []
        per_frame = Counter()
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            nm = names.get(int(c), str(c))
            boxes.append((b, int(c), float(cf)))
            cls_count[nm] += 1
            conf_sum[nm] += float(cf)
            per_frame[nm] += 1

        rows.append(dict(frame=fno, sec=round(fno / fps, 2),
                         **{f"n_{v}": per_frame.get(v, 0) for v in names.values()},
                         infer_ms=round(t_infer[-1], 1)))

        # 스냅샷은 영상 **전체에 고르게** 뽑아야 한다. 나온 순서대로 앞에서 N장을
        # 저장하면 전부 첫 몇 초에서 나와, 한 장면만 보고 모델을 판단하게 된다.
        want_snap = (per_frame.get("fallen") and n_snap < args.snapshot_fallen
                     and n_done - last_snap_at >= snap_gap)
        if writer is not None or want_snap:
            vis = draw(frame.copy(), boxes, names)
            if want_snap:
                snap_dir.mkdir(exist_ok=True)
                imwrite_u(snap_dir / f"f{fno:06d}.jpg", vis)
                n_snap += 1
                last_snap_at = n_done
            if writer is not None:
                writer.write(cv2.resize(vis, (ow, oh)))

        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done:,} 프레임 · 추론 평균 {np.mean(t_infer):.1f} ms")

    cap.release()
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - t0

    # ── 보고 ──────────────────────────────────────────────
    csv_path = out_dir / f"{vp.stem}_detections.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== 결과 · {n_done:,} 프레임 ===")
    if t_infer:
        a = np.array(t_infer)
        print(f"  추론 시간   평균 {a.mean():.1f} ms · 중앙 {np.median(a):.1f} ms · "
              f"95분위 {np.percentile(a, 95):.1f} ms")
        print(f"  추론만      {1000/a.mean():.1f} fps")
        print(f"  디코딩·저장 포함 {n_done/wall:.1f} fps  (영상 원본 {fps:.1f} fps)")
        verdict = "실시간 가능" if n_done / wall >= fps else "실시간 미달"
        print(f"  판정        {verdict}")

    print(f"\n  클래스별 탐지 수")
    tot = sum(cls_count.values())
    for nm in names.values():
        n = cls_count.get(nm, 0)
        avg = conf_sum[nm] / n if n else 0.0
        pct = n / tot * 100 if tot else 0
        n_frames = sum(1 for r in rows if r.get(f"n_{nm}", 0) > 0)
        print(f"    {nm:<10} {n:>7,}건 ({pct:>5.1f}%) · 평균 신뢰도 {avg:.3f} · "
              f"등장 프레임 {n_frames:,}/{n_done:,}")

    print(f"\n  CSV     : {csv_path}")
    if writer is not None:
        print(f"  주석영상: {out_dir / (vp.stem + '_annotated.mp4')}")
    if n_snap:
        print(f"  fallen 스냅샷 {n_snap}장: {snap_dir}")
    print("\n  ※ 이 영상에는 정답이 없다. 수치는 '무엇을 얼마나 냈나'이지 정확도가 아니다.")


if __name__ == "__main__":
    main()
