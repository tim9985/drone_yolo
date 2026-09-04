"""
okutama_verify.py — 생성된 크롭 중 라벨이 어긋난 것을 걸러낸다

왜 필요한가
  Okutama 배포 영상은 라벨 프레임과 화면이 밀려 있다. okutama_align.py 로
  영상별 오프셋을 찾아 대부분 복구했지만(1.2.10 은 0.255 → 0.590),
  **검수해 보니 여전히 일부 프레임이 어긋난다.** 한 영상 안에서도 맞는 구간과
  틀린 구간이 섞여 있어, 프레임이 여러 지점에서 빠진 것으로 보인다.
  즉 단일 오프셋으로는 완전히 못 맞춘다.

  오프셋을 구간별로 더 정교하게 추정하는 방법도 있지만, 목적을 생각하면
  **어긋난 크롭을 버리는 편이 확실하다.** 잘못 붙은 상자 하나는
  "빈 잔디가 쓰러진 사람"이라고 가르치는 것이라 학습에 독이 된다.
  프레임은 stride 를 낮추면 얼마든지 더 뽑을 수 있으니 버려도 손해가 적다.

방법
  사람 탐지기(yolov8s_stage1_all)를 각 크롭에 돌려, 라벨 상자마다
  가장 잘 겹치는 탐지 상자와의 IoU 를 구한다. 그 값이 낮은 크롭은 버린다.
  탐지기는 자세는 못 가리지만 **사람 위치는 잘 잡으므로** 정렬 검증에 쓸 수 있다.

  주의: 탐지기가 놓친 사람(가려짐 등) 때문에 멀쩡한 크롭이 버려질 수 있다.
  그래서 기준을 느슨하게 잡고(기본 0.35), 라벨 중 일정 비율만 맞으면 통과시킨다.

실행:
  python okutama_verify.py --data data/det/okutama_pose
  python okutama_verify.py --data ... --min-iou 0.4 --dry-run
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS = BASE_DIR / "weights" / "yolov8s_stage1_all.pt"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def matches(lab, det, tol=0.25):
    """라벨과 탐지가 같은 사람을 가리키는가.

    IoU 로만 보면 안 된다. 탐지기가 누운 사람의 몸통만 잡거나 한 사람을 여러
    조각으로 나눠 잡는 일이 흔해서, 라벨이 정확한데도 IoU 가 0.1 밑으로 떨어진다.
    (실측: 비치체어에 누운 사람 라벨이 정확한데 IoU 0.00 으로 버려졌다)

    대신 **중심이 상대 상자 안에 들어오는가**를 본다. 상자 크기 차이에는
    관대하면서, 정작 잡아야 할 실패 — 라벨이 빈 땅에 앉는 경우 — 는 놓치지 않는다.
    tol 은 상자를 조금 넉넉히 봐 주는 여유값이다."""
    lx1, ly1, lx2, ly2 = lab
    dx1, dy1, dx2, dy2 = det
    lw, lh = lx2 - lx1, ly2 - ly1
    dw, dh = dx2 - dx1, dy2 - dy1
    dcx, dcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
    det_in_lab = (lx1 - lw * tol <= dcx <= lx2 + lw * tol
                  and ly1 - lh * tol <= dcy <= ly2 + lh * tol)
    lab_in_det = (dx1 - dw * tol <= lcx <= dx2 + dw * tol
                  and dy1 - dh * tol <= lcy <= dy2 + dh * tol)
    return det_in_lab or lab_in_det


def read_labels(path, W, H):
    """YOLO 정규화 → 픽셀 좌표 [(cls, x1,y1,x2,y2)]"""
    out = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        q = ln.split()
        if len(q) < 5:
            continue
        c = int(q[0])
        x, y, w, h = (float(v) for v in q[1:5])
        out.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return out


def main():
    ap = argparse.ArgumentParser(description="라벨이 어긋난 크롭을 걸러낸다")
    ap.add_argument("--data", required=True, help="okutama_prep.py 산출 폴더")
    ap.add_argument("--min-iou", type=float, default=0.35,
                    help="(현재 미사용 — 중심 포함 판정으로 바꿨다)")
    ap.add_argument("--min-ratio", type=float, default=0.6,
                    help="한 크롭에서 맞은 라벨 비율이 이 값 미만이면 크롭째 버린다")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="검증용이라 확실한 탐지만 쓴다")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--strict-fallen", action="store_true", default=True,
                    help="fallen 라벨은 탐지로 확인된 것만 남긴다 (기본 켜짐)")
    ap.add_argument("--no-strict-fallen", dest="strict_fallen", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="지우지 않고 통계만")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        raise SystemExit(f"폴더가 없습니다: {root}")

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    rej_dir = root / "_rejected"
    stat = Counter()
    per_video = Counter()
    per_video_bad = Counter()

    for split in ("train", "val"):
        idir = root / "images" / split
        if not idir.is_dir():
            continue
        imgs = sorted(idir.glob("*.jpg"))
        print(f"\n[{split}] {len(imgs)}장 검증 중...")
        for n, ip in enumerate(imgs, 1):
            lp = root / "labels" / split / (ip.stem + ".txt")
            if not lp.exists():
                continue
            img = imread_u(ip)
            if img is None:
                continue
            H, W = img.shape[:2]
            labs = read_labels(lp, W, H)
            if not labs:
                continue

            r = model(img, verbose=False, imgsz=args.imgsz, conf=args.conf)[0]
            dets = [tuple(map(float, b)) for b in r.boxes.xyxy.cpu().numpy()]

            vid = ip.stem.split("_")[0]
            per_video[vid] += 1

            # **탐지 기준으로 본다.** 라벨 기준으로 보면 탐지기가 놓친 사람까지
            # "라벨이 틀렸다"로 세어, 정렬이 멀쩡한 크롭을 무더기로 버린다.
            # (실제로 그렇게 했더니 정상 영상이 70~83% 버려졌다)
            # 탐지기가 사람을 찾았는데 그 자리에 라벨이 없다면 그건 밀린 것이다.
            if not dets:
                stat["판단보류(탐지없음)"] += 1
                ok_ratio = 1.0          # 판단 근거가 없으면 살린다
            else:
                hits = sum(1 for d in dets if any(matches(lb, d) for _, *lb in labs))
                ok_ratio = hits / len(dets)

            # **fallen 라벨은 따로, 더 엄격하게 본다.**
            # 위 판정은 '탐지 → 라벨' 한 방향이라, 빈 땅에 앉은 라벨은 그 자리에
            # 탐지가 없어서 아예 검사되지 않는다. 그런데 우리가 막아야 할 실패가
            # 바로 그것이다 — 잘못 붙은 fallen 상자 하나는 "빈 잔디가 쓰러진
            # 사람"이라고 가르친다.
            # 그래서 fallen 은 **탐지로 확인된 것만** 남긴다. 탐지기의 쓰러짐
            # 재현율이 0.5~0.6 이라 맞는 것도 절반쯤 버려지지만, 남는 것은 깨끗하다.
            if args.strict_fallen:
                fallen = [lb for c, *lb in labs if c == 1]
                if fallen:
                    confirmed = sum(1 for lb in fallen if any(matches(lb, d) for d in dets))
                    if confirmed < len(fallen):
                        stat["fallen 미확인으로 버림"] += 1
                        ok_ratio = 0.0

            if ok_ratio < args.min_ratio:
                stat["버림"] += 1
                per_video_bad[vid] += 1
                stat[f"버림_{split}"] += 1
                # fallen 라벨이 들어있던 크롭인지 따로 센다 (손실이 큰 쪽)
                if any(c == 1 for c, *_ in labs):
                    stat["버림_fallen포함"] += 1
                if not args.dry_run:
                    for src, sub in ((ip, "images"), (lp, "labels")):
                        dst = rej_dir / sub / split
                        dst.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dst / src.name))
            else:
                stat["통과"] += 1
                stat[f"통과_{split}"] += 1
            if n % 200 == 0:
                print(f"    {n}/{len(imgs)}")

    total = stat["통과"] + stat["버림"]
    print(f"\n=== 검증 결과 (IoU ≥ {args.min_iou}, 통과 기준 {args.min_ratio:.0%}) ===")
    if total:
        print(f"  통과 {stat['통과']:,} ({stat['통과']/total*100:.1f}%) · "
              f"버림 {stat['버림']:,} ({stat['버림']/total*100:.1f}%)")
        print(f"  train 통과 {stat['통과_train']:,} / val 통과 {stat['통과_val']:,}")
        if stat["버림_fallen포함"]:
            print(f"  버려진 것 중 fallen 포함: {stat['버림_fallen포함']:,}")
    print(f"\n  영상별 불량률")
    for v in sorted(per_video):
        bad, tot = per_video_bad[v], per_video[v]
        bar = "█" * int(bad / max(tot, 1) * 20)
        print(f"    {v:<9} {bad:>4}/{tot:<4} {bad/max(tot,1)*100:>5.1f}%  {bar}")
    if args.dry_run:
        print("\n  (--dry-run 이라 실제로 옮기지 않았다)")
    elif stat["버림"]:
        print(f"\n  버린 것은 {rej_dir} 로 옮겼다 (지우지 않았으니 확인 가능)")


if __name__ == "__main__":
    main()
