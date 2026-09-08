"""
live_view.py — 영상에 탐지+자세 결과를 실시간으로 띄워 눈으로 판단한다

왜 필요한가
  정답셋 지표로는 안 잡히는 실패가 있다. 실제로 `ambiguous` 가 벤치·가방에
  붙는 문제는 mAP 를 아무리 봐도 안 보였고, 프레임을 눈으로 열어서야 드러났다.
  스냅샷 몇 장으로는 표본이 좁으니, 흐름을 직접 보면서 판단할 창이 필요하다.

무엇을 할 수 있나
  · 재생하면서 상자와 클래스를 실시간으로 본다
  · **신뢰도 임계값을 재생 중에 올렸다 내렸다** 하며 무엇이 사라지고 남는지 본다
  · 클래스를 켜고 꺼서 (예: ambiguous 만 숨기고) 나머지가 멀쩡한지 본다
  · 멈추고 한 프레임씩 넘기며 애매한 장면을 뜯어본다
  · 마음에 걸리는 프레임을 그 자리에서 저장한다

조작
  space   재생 / 일시정지
  d , →   한 프레임 앞으로   (일시정지 상태에서)
  a , ←   한 프레임 뒤로     (일시정지 상태에서 · 되감기는 느리다)
  + / -   신뢰도 임계값 ±0.05
  1 2 3   person / fallen / ambiguous 표시 토글
  f       fallen 이 있는 다음 프레임으로 건너뛰기
  s       현재 화면 저장
  q, ESC  종료

입력
  --source 는 영상 파일 · 웹캠 번호(0) · RTSP/HTTP 주소를 다 받는다.
  나중에 실기체 스트림을 붙일 때도 그대로 쓴다.

실행
  python live_view.py --source data/raw/okutama/Drone2/Noon/2.2.10.mp4
  python live_view.py --source data/raw/okutama/Drone1/Morning/1.1.9.mp4 --conf 0.35
"""
import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# BGR. fallen 은 놓치면 안 되는 대상이라 빨강, ambiguous 는 보류를 뜻하는 노랑
COLORS = {"person": (90, 190, 90), "fallen": (60, 60, 235), "ambiguous": (40, 190, 235)}
DEFAULT_COLOR = (200, 200, 200)
WIN = "drone vision - live"


def imwrite_u(path, img):
    """한글 경로에서 cv2.imwrite 는 실패한다."""
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def put(img, text, org, color=(255, 255, 255), scale=0.5, thick=1, bg=True):
    if bg:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="탐지+자세 실시간 확인 창")
    ap.add_argument("--source", required=True,
                    help="영상 파일 · 웹캠 번호(0) · RTSP/HTTP 주소")
    ap.add_argument("--weights", default="weights/yolov8s_pose3_sn_freeze.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=1600, help="창 가로 크기")
    ap.add_argument("--stride", type=int, default=1, help="N 프레임마다 1장 추론")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, str) and not src.startswith(("rtsp", "http")) \
            and not Path(src).exists():
        raise SystemExit(f"입력을 찾을 수 없습니다: {src}")

    save_dir = Path(args.save_dir) if args.save_dir else (BASE_DIR / "runs_video" / "_saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    # 토글용 — 클래스 이름 → 표시 여부
    show = {v: True for v in names.values()}

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {src}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"입력  : {src}")
    print(f"영상  : {W}x{H} · {src_fps:.1f} fps · {n_total:,} 프레임" if n_total > 0
          else f"영상  : {W}x{H} · 스트림")
    print(f"모델  : {Path(args.weights).name} · {list(names.values())}")
    print("\n조작: space 재생/정지 · a/d 한 프레임 · +/- 임계값 · 1/2/3 클래스 토글")
    print("      f 다음 fallen · s 저장 · q 종료\n")

    conf = args.conf
    paused = False
    fno = -1
    seek_fallen = False
    frame = None
    boxes = []
    fps_hist = deque(maxlen=30)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.width, int(args.width * H / max(W, 1)))

    def infer(img):
        t = time.perf_counter()
        r = model(img, verbose=False, imgsz=args.imgsz, conf=0.05,
                  device=args.device)[0]
        dt = (time.perf_counter() - t) * 1000
        out = []
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            out.append((b, names.get(int(c), str(c)), float(cf)))
        return out, dt

    # 추론은 conf 0.05 로 한 번만 하고, 화면 임계값은 그 결과를 걸러서 쓴다.
    # 그래야 임계값을 바꿀 때마다 재추론하지 않고 즉시 반응한다.
    dt_ms = 0.0
    while True:
        if not paused or frame is None:
            ok, f = cap.read()
            if not ok:
                if n_total > 0:
                    print("영상 끝")
                    break
                continue
            fno += 1
            if args.stride > 1 and fno % args.stride:
                continue
            frame = f
            boxes, dt_ms = infer(frame)
            fps_hist.append(dt_ms)

            if seek_fallen:
                if any(nm == "fallen" and cf >= conf for _b, nm, cf in boxes):
                    seek_fallen = False
                    paused = True
                else:
                    continue

        vis = frame.copy()
        shown = Counter()
        for b, nm, cf in boxes:
            if cf < conf or not show.get(nm, True):
                continue
            shown[nm] += 1
            x1, y1, x2, y2 = (int(v) for v in b)
            col = COLORS.get(nm, DEFAULT_COLOR)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if nm == "fallen" else 2)
            put(vis, f"{nm} {cf:.2f}", (x1 + 2, y1 - 5), col, 0.5, 1)

        # ── 상태 표시줄 ──
        bar = 30
        pos = f"{fno:,}/{n_total:,}" if n_total > 0 else f"{fno:,}"
        avg = np.mean(fps_hist) if fps_hist else 0
        put(vis, f"frame {pos}  |  {fno/src_fps:6.1f}s  |  infer {dt_ms:5.1f}ms "
                 f"({1000/max(avg,1e-6):4.1f} fps)", (12, bar), (255, 255, 255), 0.6, 2)
        bar += 26
        put(vis, f"conf >= {conf:.2f}   [{'PAUSED' if paused else 'PLAY'}]",
            (12, bar), (0, 255, 255) if paused else (255, 255, 255), 0.6, 2)
        bar += 26
        for i, nm in enumerate(names.values(), start=1):
            mark = "on " if show.get(nm, True) else "off"
            put(vis, f"[{i}] {nm} {shown.get(nm,0)} ({mark})", (12, bar),
                COLORS.get(nm, DEFAULT_COLOR), 0.6, 2)
            bar += 24

        cv2.imshow(WIN, vis)
        # 일시정지 중에는 키를 기다린다. 재생 중에는 원본 속도에 맞춰 잠깐만 기다린다.
        wait = 0 if paused else max(1, int(1000 / src_fps - dt_ms))
        k = cv2.waitKey(wait) & 0xFF

        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k in (ord("d"), 83):           # → 한 프레임 앞으로
            paused = True
            ok, f = cap.read()
            if ok:
                fno += 1
                frame = f
                boxes, dt_ms = infer(frame)
        elif k in (ord("a"), 81):           # ← 한 프레임 뒤로
            paused = True
            tgt = max(0, fno - 1)
            # H.264 는 임의 위치 탐색이 부정확하다. 앞 키프레임부터 다시 읽는다.
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, tgt - 30))
            cur = max(0, tgt - 30)
            while cur < tgt:
                ok, f = cap.read()
                if not ok:
                    break
                cur += 1
            ok, f = cap.read()
            if ok:
                fno = tgt
                frame = f
                boxes, dt_ms = infer(frame)
        elif k in (ord("+"), ord("=")):
            conf = min(0.95, round(conf + 0.05, 2))
        elif k in (ord("-"), ord("_")):
            conf = max(0.05, round(conf - 0.05, 2))
        elif k in (ord("1"), ord("2"), ord("3")):
            idx = k - ord("1")
            vals = list(names.values())
            if idx < len(vals):
                show[vals[idx]] = not show[vals[idx]]
        elif k == ord("f"):
            seek_fallen = True
            paused = False
        elif k == ord("s"):
            p = save_dir / f"frame_{fno:06d}_conf{int(conf*100)}.jpg"
            imwrite_u(p, vis)
            print(f"저장: {p}")

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
