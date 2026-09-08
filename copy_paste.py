"""
copy_paste.py — 오려낸 사람을 다른 배경에 합성해 학습셋을 늘린다

왜 필요한가
  쓰러진 사람 데이터가 **한 곳에 몰려 있다.**
      SARD    슬로베니아 훈련장 · 배우 소수
      Okutama 일본 공원 · 배우 10명 미만
  둘 다 배우와 장소가 좁아, 모델이 자세가 아니라 **그 사람·그 잔디**를 외울 수 있다.
  실제로 nomad30 은 한 도메인만 학습해 WiSARD 가 0.544 → 0.242 로 무너졌다.

  Copy-Paste(Ghiasi 외, CVPR 2021)는 인스턴스를 다른 배경에 옮겨 붙여
  **"자세 × 배경" 조합을 늘린다.** 자세는 그대로 두고 배경만 바꾸므로
  모델이 배경에 기대지 못하게 만든다.

배경 우선순위 (우리 운용 환경 기준)
  1순위 NOMAD 농장   — 농촌·들판·나무. **지형이 우리 시연 환경과 가장 가깝다**
  2순위 WiSARD 9월   — 마른 갈색 식생·암반. 색조가 늦가을에 가깝다
  3순위 WiSARD 1월   — 폭설. 한국 11~12월과 다르므로 **소량만** 강건성 보험으로

  한국 늦가을(마른 갈색 풀·잎 진 나무)과 정확히 맞는 데이터는 우리에게 없다.
  그래서 --autumn 으로 배경 채도를 낮추고 갈색 쪽으로 밀어 간극을 좁힌다.

붙일 때 지키는 것
  · **크기** — 배경 이미지의 기존 사람 상자에서 중앙 크기를 재서 거기 맞춘다.
    고도를 몰라도 되고 GSD 계산보다 튼튼하다.
  · **겹침 회피** — 기존 상자와 IoU 가 겹치지 않는 자리에만 놓는다.
  · **그림자** — 하향 시점에서 그림자 없는 사람은 떠 보인다. 약한 타원을 깐다.
  · **가장자리** — 알파를 흐려 딱 잘린 티를 줄인다.

실행
  python copy_paste.py --bank runs_instances/sard_fallen --n 1500
  python copy_paste.py --bank runs_instances/sard_fallen --n 800 --autumn 0.5
"""
import argparse
import json
import random
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
CLASS_NAMES = ["person", "fallen", "ambiguous"]
PERSON, FALLEN, AMBIG = 0, 1, 2

# 배경 후보. (이름, 이미지폴더, 라벨폴더 or None, 기본 가중치)
#   pose3_sn 은 NOMAD+SARD 가 이미 3클래스로 붙어 있어 그대로 쓸 수 있다.
BG_SOURCES = [
    ("NOMAD",       "data/det/pose3_sn/images/train", "data/det/pose3_sn/labels/train", 6, "nomad_"),
    ("WiSARD 9월",  "data/det/wisard/images/train",   "data/det/wisard/labels/train",   3, None),
    ("WiSARD 1월",  "data/det/wisard/images/train",   "data/det/wisard/labels/train",   1, "DJI_0582"),
]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(Path(p).suffix, img)
    if ok:
        buf.tofile(str(p))
    return ok


def read_yolo(lp, W, H):
    """(cls, x1,y1,x2,y2) 목록"""
    out = []
    if not lp or not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        c = int(v[0])
        x, y, w, h = (float(t) for t in v[1:5])
        out.append((c, (x - w / 2) * W, (y - h / 2) * H,
                    (x + w / 2) * W, (y + h / 2) * H))
    return out


def overlaps(box, others, margin=6):
    x1, y1, x2, y2 = box
    for _c, ox1, oy1, ox2, oy2 in others:
        if not (x2 + margin < ox1 or x1 - margin > ox2
                or y2 + margin < oy1 or y1 - margin > oy2):
            return True
    return False


def autumn_shift(img, strength=1.0):
    """여름 초록을 늦가을 마른 들판 쪽으로 민다.

    한국 11~12월 농촌은 마른 갈색 풀과 잎 진 나무다. NOMAD 는 짙은 초록이라
    그대로 쓰면 색이 안 맞는다. 채도를 낮추고 색상을 황갈색 쪽으로 옮긴다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 초록(H 35~85)을 황갈색(H 15~30) 쪽으로
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90)
    hsv[..., 0] = np.where(green,
                           hsv[..., 0] - 18 * strength, hsv[..., 0])
    hsv[..., 1] *= (1 - 0.42 * strength)          # 채도 down
    hsv[..., 2] *= (1 - 0.06 * strength)          # 살짝 어둡게
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste(bg, inst, cx, cy, shadow_dx, shadow_dy, shadow_a=0.30):
    """RGBA 인스턴스를 bg 의 (cx,cy) 중심에 합성. 그림자 먼저, 사람 나중."""
    ih, iw = inst.shape[:2]
    x1, y1 = int(cx - iw / 2), int(cy - ih / 2)
    H, W = bg.shape[:2]
    if x1 < 0 or y1 < 0 or x1 + iw > W or y1 + ih > H:
        return None
    a = (inst[:, :, 3:4].astype(np.float32) / 255.0)

    # ── 그림자 — 알파를 흐려 옆으로 민 어두운 자국
    sh = cv2.GaussianBlur(inst[:, :, 3], (0, 0), max(iw, ih) * 0.10)
    sx1, sy1 = x1 + shadow_dx, y1 + shadow_dy
    if 0 <= sx1 and 0 <= sy1 and sx1 + iw <= W and sy1 + ih <= H:
        roi = bg[sy1:sy1 + ih, sx1:sx1 + iw].astype(np.float32)
        sa = (sh.astype(np.float32) / 255.0 * shadow_a)[:, :, None]
        bg[sy1:sy1 + ih, sx1:sx1 + iw] = (roi * (1 - sa)).astype(np.uint8)

    # ── 사람
    roi = bg[y1:y1 + ih, x1:x1 + iw].astype(np.float32)
    out = inst[:, :, :3].astype(np.float32) * a + roi * (1 - a)
    bg[y1:y1 + ih, x1:x1 + iw] = out.astype(np.uint8)
    return (x1, y1, x1 + iw, y1 + ih)


def main():
    ap = argparse.ArgumentParser(description="Copy-Paste 합성 학습셋 생성")
    ap.add_argument("--bank", required=True, help="extract_instances.py 출력 폴더")
    ap.add_argument("--n", type=int, default=1200, help="만들 이미지 수")
    ap.add_argument("--paste-min", type=int, default=1)
    ap.add_argument("--paste-max", type=int, default=3)
    ap.add_argument("--classes", nargs="+", default=["fallen"],
                    help="붙일 인스턴스 클래스")
    ap.add_argument("--autumn", type=float, default=0.45,
                    help="늦가을 색 변환을 적용할 확률 (0~1)")
    ap.add_argument("--scale-jitter", type=float, default=0.18)
    ap.add_argument("--rotate", type=float, default=180.0,
                    help="회전 범위(±도). 하향 시점이라 사람 방향은 자유롭다")
    ap.add_argument("--target-px", type=int, default=0,
                    help="배경에 기존 사람이 없을 때 쓸 목표 크기. 0이면 자동")
    ap.add_argument("--out", default="data/det/cp_fallen")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bank_dir = Path(args.bank)
    meta = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    meta = [m for m in meta if m["cls"] in args.classes]
    if not meta:
        raise SystemExit(f"은행에 {args.classes} 인스턴스가 없습니다: {bank_dir}")

    insts = []
    for m in meta:
        im = imread_u(bank_dir / m["file"])
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        insts.append((im, m["cls"]))
    if not insts:
        raise SystemExit("읽을 수 있는 인스턴스가 없습니다")
    print(f"인스턴스 {len(insts):,}개 ({args.classes})")

    # ── 배경 수집 ────────────────────────────────────────────
    bgs = []      # (이름, 이미지경로, 라벨경로)
    for nm, idir, ldir, weight, prefix in BG_SOURCES:
        ip = BASE_DIR / idir
        if not ip.is_dir():
            print(f"  건너뜀 {nm} — 없음")
            continue
        files = sorted(list(ip.glob("*.jpg")) + list(ip.glob("*.png")))
        if prefix:
            if prefix == "DJI_0582":
                files = [f for f in files if f.name.startswith(prefix)]
            else:
                files = [f for f in files if f.name.startswith(prefix)]
        else:
            files = [f for f in files if not f.name.startswith("DJI_0582")]
        if not files:
            continue
        lp = BASE_DIR / ldir if ldir else None
        pool = [(nm, f, (lp / (f.stem + ".txt")) if lp else None) for f in files]
        # 가중치만큼 반복해 넣어 선택 확률을 조절한다
        bgs += pool * weight
        print(f"  배경 {nm:<10} {len(files):>6,}장 · 가중치 {weight}")
    if not bgs:
        raise SystemExit("배경이 없습니다")
    rng.shuffle(bgs)

    out = Path(args.out).resolve()
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stat = {sp: Counter() for sp in ("train", "val")}
    src_used = Counter()
    fails = Counter()
    made = 0
    bi = 0

    while made < args.n and bi < len(bgs) * 3:
        nm, ipath, lpath = bgs[bi % len(bgs)]
        bi += 1
        bg = imread_u(ipath)
        if bg is None or bg.ndim != 3:
            fails["배경 못 읽음"] += 1
            continue
        bg = bg[:, :, :3].copy()
        H, W = bg.shape[:2]
        existing = read_yolo(lpath, W, H)

        # 크기 기준 — 기존 사람 상자의 중앙 장변
        if existing:
            ref = float(np.median([max(b[3] - b[1], b[4] - b[2]) for b in existing]))
        elif args.target_px:
            ref = float(args.target_px)
        else:
            fails["크기 기준 없음"] += 1
            continue
        if ref < 12:
            fails["기준 너무 작음"] += 1
            continue

        if rng.random() < args.autumn:
            bg = autumn_shift(bg, strength=rng.uniform(0.6, 1.0))

        boxes = list(existing)
        added = 0
        for _ in range(rng.randint(args.paste_min, args.paste_max)):
            inst, cls = insts[rng.randrange(len(insts))]
            # 회전 — 하향 시점이라 누운 방향은 어느 쪽이든 자연스럽다
            if args.rotate:
                ang = rng.uniform(-args.rotate, args.rotate)
                ih, iw = inst.shape[:2]
                M = cv2.getRotationMatrix2D((iw / 2, ih / 2), ang, 1.0)
                cos, sin = abs(M[0, 0]), abs(M[0, 1])
                nw, nh = int(ih * sin + iw * cos), int(ih * cos + iw * sin)
                M[0, 2] += nw / 2 - iw / 2
                M[1, 2] += nh / 2 - ih / 2
                inst = cv2.warpAffine(inst, M, (nw, nh), flags=cv2.INTER_LINEAR,
                                      borderValue=(0, 0, 0, 0))
            ih, iw = inst.shape[:2]
            # 쓰러진 사람은 서 있는 사람보다 길다. 기준(선 자세)의 2~3배로 둔다.
            want = ref * rng.uniform(2.0, 3.0) if cls == "fallen" else ref
            want *= rng.uniform(1 - args.scale_jitter, 1 + args.scale_jitter)
            s = want / max(iw, ih)
            nw, nh = max(6, int(iw * s)), max(6, int(ih * s))
            if nw >= W - 8 or nh >= H - 8:
                continue
            piece = cv2.resize(inst, (nw, nh), interpolation=cv2.INTER_AREA)
            # 알파 가장자리를 한 번 더 부드럽게
            piece[:, :, 3] = cv2.GaussianBlur(piece[:, :, 3], (3, 3), 0)

            placed = None
            for _try in range(24):
                cx = rng.randint(nw // 2 + 2, W - nw // 2 - 2)
                cy = rng.randint(nh // 2 + 2, H - nh // 2 - 2)
                cand = (cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)
                if overlaps(cand, boxes):
                    continue
                sdx = int(rng.choice([-1, 1]) * nw * rng.uniform(0.05, 0.16))
                sdy = int(rng.choice([-1, 1]) * nh * rng.uniform(0.05, 0.16))
                placed = paste(bg, piece, cx, cy, sdx, sdy)
                break
            if placed is None:
                continue
            ci = FALLEN if cls == "fallen" else PERSON
            boxes.append((ci, *placed))
            added += 1

        if added == 0:
            fails["붙일 자리 없음"] += 1
            continue

        split = "val" if rng.random() < args.val_frac else "train"
        stem = f"cp_{made:06d}"
        lines = []
        for c, x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{bw/W:.6f} {bh/H:.6f}")
            stat[split][c] += 1
        if not lines:
            continue
        imwrite_u(out / "images" / split / f"{stem}.jpg", bg)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        stat[split]["images"] += 1
        src_used[nm] += 1
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.n}")

    yml = BASE_DIR / "configs" / f"data_{out.name}.yaml"
    yml.write_text(
        "# Copy-Paste 합성 — 0 person / 1 fallen / 2 ambiguous\n"
        f"# 인스턴스 {bank_dir.name} · 늦가을 색변환 {args.autumn}\n"
        f"train: {(out / 'images' / 'train').as_posix()}\n"
        f"val: {(out / 'images' / 'val').as_posix()}\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n", encoding="utf-8")

    print(f"\n=== Copy-Paste 합성 ===")
    print(f"{'분할':<7}{'이미지':>8}{'person':>10}{'fallen':>9}{'ambiguous':>12}")
    for sp in ("train", "val"):
        c = stat[sp]
        if c["images"]:
            print(f"  {sp:<5}{c['images']:>8,}{c[PERSON]:>10,}{c[FALLEN]:>9,}{c[AMBIG]:>12,}")
    print(f"\n  배경 사용: {dict(src_used)}")
    if fails:
        print(f"  실패: {dict(fails)}")
    print(f"\n  설정: {yml}\n  출력: {out}")
    print("  ※ 학습 전에 label_review.py 로 눈 검수할 것 —"
          " 어색한 합성이 섞이면 모델이 '붙인 자국'을 학습한다")


if __name__ == "__main__":
    main()
