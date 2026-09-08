"""
label_review.py — 자세 라벨을 눈으로 검수한다

왜 필요한가
  자세 라벨은 **틀려도 지표에 안 나타난다.** 모델은 틀린 정답에 맞춰 잘 학습되고
  검증셋도 같은 기준으로 틀려 있으니 mAP 는 멀쩡하게 나온다. NOMAD 의 전환 동작
  오염도 사람이 직접 눈으로 보고서야 발견했다. 학습에 넣기 전에 봐야 한다.

무엇을 만드나
  1) 클래스별 **조각 모음판** (기본) — 라벨 상자만 잘라 격자로 붙인다.
     100개를 한 화면에서 훑을 수 있어 "이게 왜 fallen 이지?" 를 빨리 찾는다.
     각 조각에 출처 파일과 상자 크기를 적어 두므로 원본을 바로 열 수 있다.
  2) **전체 프레임 주석본** (--frames) — 맥락이 필요할 때. 상자와 클래스를 그린다.

  모델 예측이 아니라 **정답 라벨**을 그린다. 검수 대상이 라벨이기 때문이다.

실행
  python label_review.py --data data/det/okutama3 --split train
  python label_review.py --data data/det/okutama3 --split train --cls fallen --per-sheet 100
  python label_review.py --data data/det/okutama3 --frames 12
출력: runs_review/<데이터셋>_<분할>/
"""
import argparse
import csv
import random
import sys
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
CLASS_NAMES = ["person", "fallen", "ambiguous"]
COLORS = {"person": (90, 190, 90), "fallen": (60, 60, 235), "ambiguous": (40, 190, 235)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(Path(p).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(p))
    return ok


def read_labels(lp):
    out = []
    for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        out.append((int(v[0]), *[float(x) for x in v[1:5]]))
    return out


def crop_box(img, cx, cy, bw, bh, pad=0.35, size=132):
    """상자를 여유 있게 잘라 정사각 타일로 만든다. 여유를 줘야 자세가 보인다."""
    H, W = img.shape[:2]
    x, y = cx * W, cy * H
    w, h = bw * W, bh * H
    s = max(w, h) * (1 + pad * 2)
    x1, y1 = int(x - s / 2), int(y - s / 2)
    x2, y2 = int(x + s / 2), int(y + s / 2)
    # 화면 밖은 검게 채운다. 잘라서 비율이 깨지면 자세 판단이 왜곡된다.
    tile = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x2), min(H, y2)
    if sx2 > sx1 and sy2 > sy1:
        tile[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_CUBIC)
    # 원 상자 위치를 타일 안에 표시 — 어느 것을 보라는 건지 분명해진다
    m = int(size * pad / (1 + pad * 2))
    cv2.rectangle(tile, (m, m), (size - m, size - m), (255, 255, 255), 1)
    return tile


def make_sheet(tiles, cols, size, header, cap_h=26):
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (size + cap_h) + 34, cols * size, 3), 22, np.uint8)
    cv2.putText(sheet, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, (tile, cap) in enumerate(tiles):
        r, c = divmod(i, cols)
        y = 34 + r * (size + cap_h)
        x = c * size
        sheet[y:y + size, x:x + size] = tile
        cv2.putText(sheet, cap[:22], (x + 3, y + size + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
    return sheet


def main():
    ap = argparse.ArgumentParser(description="자세 라벨 눈 검수")
    ap.add_argument("--data", required=True, help="데이터셋 폴더 (images/labels 를 품은)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--cls", default=None,
                    help="특정 클래스만 (person/fallen/ambiguous). 생략하면 전부")
    ap.add_argument("--per-sheet", type=int, default=80, help="모음판 한 장에 담을 조각 수")
    ap.add_argument("--max-sheets", type=int, default=4, help="클래스당 모음판 장수")
    ap.add_argument("--size", type=int, default=132, help="조각 한 변 픽셀")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--frames", type=int, default=0,
                    help="전체 프레임 주석본을 N 장 저장 (맥락 확인용)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.data).resolve()
    idir = root / "images" / args.split
    ldir = root / "labels" / args.split
    if not idir.is_dir():
        raise SystemExit(f"없는 경로: {idir}")

    out = Path(args.out) if args.out else BASE_DIR / "runs_review" / f"{root.name}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(idir.glob("*.jpg")) + list(idir.glob("*.png")))
    print(f"{root.name} / {args.split} — 이미지 {len(imgs):,}장")

    # ── 색인부터 만든다. 어떤 파일에 어떤 클래스가 몇 개인지 ──
    index = []
    by_cls = defaultdict(list)      # cls → [(이미지, 상자), ...]
    total = Counter()
    for ip in imgs:
        lp = ldir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        labs = read_labels(lp)
        cnt = Counter(l[0] for l in labs)
        total.update(cnt)
        index.append(dict(image=ip.name,
                          **{f"n_{n}": cnt.get(i, 0) for i, n in enumerate(CLASS_NAMES)},
                          total=len(labs)))
        for l in labs:
            by_cls[l[0]].append((ip, l))

    csv_path = out / "index.csv"
    if index:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
            w.writeheader()
            w.writerows(index)

    print(f"\n{'클래스':<12}{'상자 수':>9}{'비중':>8}{'등장 이미지':>12}")
    tot = sum(total.values())
    for i, nm in enumerate(CLASS_NAMES):
        n = total.get(i, 0)
        n_img = sum(1 for r in index if r[f"n_{nm}"] > 0)
        print(f"  {nm:<10}{n:>9,}{n/tot*100 if tot else 0:>7.1f}%{n_img:>12,}")

    # ── 클래스별 조각 모음판 ──
    rng = random.Random(args.seed)
    targets = [args.cls] if args.cls else CLASS_NAMES
    for nm in targets:
        ci = CLASS_NAMES.index(nm)
        items = by_cls.get(ci, [])
        if not items:
            print(f"\n  {nm}: 상자 없음 — 건너뜀")
            continue
        rng.shuffle(items)              # 앞쪽 영상에 몰리지 않게 섞는다
        n_sheet = min(args.max_sheets,
                      (len(items) + args.per_sheet - 1) // args.per_sheet)
        for s in range(n_sheet):
            chunk = items[s * args.per_sheet:(s + 1) * args.per_sheet]
            tiles = []
            for ip, (_c, cx, cy, bw, bh) in chunk:
                img = imread_u(ip)
                if img is None:
                    continue
                H, W = img.shape[:2]
                px = int(max(bw * W, bh * H))
                tiles.append((crop_box(img, cx, cy, bw, bh, size=args.size),
                              f"{ip.stem.replace('oku_','')} {px}px"))
            if not tiles:
                continue
            hdr = (f"{nm.upper()}  [{s+1}/{n_sheet}]  {len(tiles)}/{len(items):,}개  "
                   f"· {root.name}/{args.split} · 흰 테두리 안이 라벨 상자")
            sheet = make_sheet(tiles, args.cols, args.size, hdr)
            p = out / f"sheet_{nm}_{s+1}.jpg"
            imwrite_u(p, sheet)
            print(f"  저장 {p.name}  ({len(tiles)}개)")

    # ── 전체 프레임 주석본 ──
    if args.frames:
        fdir = out / "frames"
        fdir.mkdir(exist_ok=True)
        # fallen 이 있는 프레임을 우선 보여준다
        cand = [r for r in index if r["n_fallen"] > 0] or index
        rng.shuffle(cand)
        for r in cand[:args.frames]:
            ip = idir / r["image"]
            img = imread_u(ip)
            if img is None:
                continue
            H, W = img.shape[:2]
            for c, cx, cy, bw, bh in read_labels(ldir / (ip.stem + ".txt")):
                nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
                x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                col = COLORS.get(nm, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if nm == "fallen" else 2)
                cv2.putText(img, nm, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, 2, cv2.LINE_AA)
            imwrite_u(fdir / f"{ip.stem}.jpg", img)
        print(f"  전체 프레임 {min(args.frames, len(cand))}장: {fdir}")

    print(f"\n색인: {csv_path}\n출력: {out}")


if __name__ == "__main__":
    main()
