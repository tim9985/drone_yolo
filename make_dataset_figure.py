"""
make_dataset_figure.py — 발표용 데이터셋 구성 시각자료 생성

왜 필요한가
  "NOMAD 7,624장 + WiSARD 6,718장" 같은 숫자만으로는 청중이 데이터가 어떻게 생겼는지
  알 수 없다. 계절·지형·자세가 실제로 얼마나 다른지는 한 장의 그림이 표 열 줄보다 낫다.

만드는 것
  4행 × 3열 격자. 각 행이 하나의 데이터 출처이며, 정답 상자를 그려 넣어
  "라벨이 붙은 학습 데이터"임을 보여준다.

  NOMAD      여름 · 미국 농장 · 배우 30명 (활동 라벨 보유)
  WiSARD 9월  가을 · 산지
  WiSARD 1월  겨울 · 설경        ← 최종 시현 환경에 가장 가까움
  합성        AirSim 자동 생성 · 쓰러진 자세 (수작업 라벨링 0장)

실행: python make_dataset_figure.py [--out <경로>] [--seed 7]
"""
import argparse
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = Path(r"C:\Users\timjj\Desktop\발표 자료\02_데이터셋\데이터셋_구성_한눈에.png")

FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"

TILE_W, TILE_H = 420, 236        # 16:9 유지 (원본 크롭이 1280x720)
LABEL_W = 260                     # 좌측 설명 칸
PAD = 14
BOX_COLOR = (60, 220, 90)         # 정답 상자 (BGR 아님 — PIL RGB)

# 행 정의: (제목, 부제, 이미지 폴더, 파일 필터)
ROWS = [
    ("NOMAD", "여름 · 미국 농장 · 배우 30명\n활동 라벨(보행·은폐·쓰러짐) 보유",
     "data/dataset_nomad/images/train", None),
    ("WiSARD 9월", "가을 · 산지\n프로젝트 시기와 근접",
     "data/dataset_wisard/images/train", lambda p: not p.name.startswith("DJI_0582")),
    ("WiSARD 1월", "겨울 · 설경\n최종 시현 환경에 가장 가까움",
     "data/dataset_wisard/images/train", lambda p: p.name.startswith("DJI_0582")),
    ("합성 (AirSim)", "쓰러진 자세 · 잔해 가림\n수작업 라벨링 0장 — 자동 생성",
     "data/dataset_synth/yolo/images/train", None),
]


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def pick(folder, n, pred, rng):
    """라벨이 비어 있지 않은 표본만 고른다. 빈 라벨을 쓰면 '상자 없는 학습 데이터'로 보인다."""
    paths = []
    for ext in ("*.jpg", "*.png"):
        paths += sorted((BASE_DIR / folder).glob(ext))
    cand = []
    for p in paths:
        lp = Path(str(p).replace("images", "labels")).with_suffix(".txt")
        if not lp.exists():
            continue
        lines = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue
        if pred and not pred(p):
            continue
        cand.append((p, lines))
    rng.shuffle(cand)
    return cand[:n]


def render_tile(img_path, lines):
    """이미지에 정답 상자를 그려 타일 크기로 반환 (PIL RGB)."""
    img = imread_u(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for ln in lines:
        v = ln.split()
        if len(v) < 5:
            continue
        cx, cy, bw, bh = [float(x) for x in v[1:5]]
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        # 작은 대상이라 선을 얇게 하면 축소 후 사라진다
        d.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=max(3, int(w / 320)))
    return pil.resize((TILE_W, TILE_H), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    try:
        f_title = ImageFont.truetype(FONT_BOLD, 21)
        f_sub = ImageFont.truetype(FONT_PATH, 14)
        f_cap = ImageFont.truetype(FONT_PATH, 13)
        f_head = ImageFont.truetype(FONT_BOLD, 25)
    except OSError:
        raise SystemExit(f"한글 글꼴을 찾지 못했습니다: {FONT_PATH}")

    n_col = 3
    head_h = 62
    W = LABEL_W + n_col * (TILE_W + PAD) + PAD
    H = head_h + len(ROWS) * (TILE_H + PAD) + PAD + 26

    canvas = Image.new("RGB", (W, H), (247, 248, 247))
    d = ImageDraw.Draw(canvas)
    d.text((PAD + 6, 16), "학습 데이터 구성 — 계절 · 지형 · 자세", font=f_head, fill=(17, 24, 26))
    d.line([(PAD, head_h - 10), (W - PAD, head_h - 10)], fill=(182, 194, 192), width=2)

    total = 0
    for r, (title, sub, folder, pred) in enumerate(ROWS):
        y = head_h + r * (TILE_H + PAD)
        d.text((PAD + 6, y + 6), title, font=f_title, fill=(21, 94, 99))
        for i, line in enumerate(sub.split("\n")):
            d.text((PAD + 6, y + 38 + i * 20), line, font=f_sub, fill=(71, 85, 90))

        got = pick(folder, n_col, pred, rng)
        for c in range(n_col):
            x = LABEL_W + c * (TILE_W + PAD)
            if c < len(got):
                p, lines = got[c]
                tile = render_tile(p, lines)
                if tile is not None:
                    canvas.paste(tile, (x, y))
                    d.rectangle([x, y, x + TILE_W, y + TILE_H],
                                outline=(200, 208, 206), width=1)
                    total += len(lines)
            else:
                d.rectangle([x, y, x + TILE_W, y + TILE_H],
                            outline=(210, 216, 214), width=1)

    d.text((PAD + 6, H - 24),
           "초록 상자 = 학습에 사용된 정답 라벨   ·   모든 이미지는 운용 조건(고도 20m·화각 60°·1280×720)에 맞춰 정규화",
           font=f_cap, fill=(119, 134, 139))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"저장: {out}  ({W}x{H}, 표시된 정답 상자 {total}개)")


if __name__ == "__main__":
    main()
