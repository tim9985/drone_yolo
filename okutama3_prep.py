"""
okutama3_prep.py — Okutama 원본 영상에서 자세 3클래스 학습셋을 만든다

왜 새로 만드는가
  okutama_prep.py 는 "관측 해상도가 부족해서 자세 분류가 안 되는가"를 검증하려고
  사람이 200 px 이 되도록 **확대해서 크롭**했다(6차 시도). 그 가설은 기각됐고,
  이제 목적이 다르다. 지금 필요한 것은 **우리 운용 조건 그대로의 영상**이다.

우리 조건과 Okutama 가 정확히 맞는다
  기준서: 54° HFOV · 1080p · 고도 16 m → 선 자세 **59 px**
  Okutama 원본 라벨 상자 장변 중앙값 118 px (3840 폭) → 1920 폭 환산 **59 px**

  그래서 이 스크립트는 **자르지도 확대하지도 않는다.** 4K 프레임을 1920x1080 으로
  줄이기만 한다. 그러면 실기체 카메라가 내놓을 그림과 사람 크기가 같아진다.
  43편 중 36편이 환산 고도 16~30 m 구간에 든다.

라벨 품질
  Okutama 는 **프레임별·상자별 사람 주석**이다. NOMAD 처럼 활동 구간에서
  변환한 것이 아니라 SARD 와 같은 급이다. Lying 이 40편에 11,922개 있다 —
  SARD laying_down(1,096)의 10배다.

클래스 매핑
  Lying                                  → fallen(1)
  Standing · Walking · Sitting · Running → person(0)
  행동 라벨 없음                          → ambiguous(2)

  마지막 항목이 중요하다. 원 주석자가 자세를 특정하지 않은 상자이므로
  "애매함"의 정의가 **한 출처 안에서 일관된다.** SARD not_defined 를 섞었을 때는
  가려짐·과소 크기까지 뒤섞여 모델이 "흐릿한 덩어리 = ambiguous" 를 배웠고,
  실영상에서 벤치·가방에 상자가 붙었다.

프레임 정렬
  영상과 라벨의 프레임 번호가 영상마다 0~21 프레임 어긋나 있다.
  configs/okutama_offsets.json 의 값을 반드시 적용한다. 없으면 그 영상은 건너뛴다 —
  어긋난 채로 넣으면 상자가 사람에서 빗나간 채 학습된다.

  탐색은 순차 디코딩만 쓴다. H.264 에서 cap.set(POS_FRAMES) 로 건너뛰면
  프레임이 어긋난다(이미 겪은 문제).

분할
  **영상 단위**로 나눈다. 프레임 단위로 섞으면 연속 프레임 상관 때문에
  성능이 부풀려진다.

실행
  python okutama3_prep.py --videos 2.2.10 1.2.10 2.2.6 ... --stride 15
"""
import argparse
import json
import re
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
SRC = BASE_DIR / "data" / "raw" / "okutama"
LAB_DIR = SRC / "Labels" / "MultiActionLabels" / "3840x2160"
OFFSETS_JSON = BASE_DIR / "configs" / "okutama_offsets.json"
OUT = BASE_DIR / "data" / "det" / "okutama3"

OUT_W, OUT_H = 1920, 1080          # 실기체 카메라 해상도
CLASS_NAMES = ["person", "fallen", "ambiguous"]
PERSON, FALLEN, AMBIG = 0, 1, 2
POSE_TO_CLS = {"Lying": FALLEN, "Standing": PERSON, "Walking": PERSON,
               "Sitting": PERSON, "Running": PERSON}


def parse_labels(path):
    """프레임 → [(cls, x1, y1, x2, y2), ...]   좌표는 3840x2160 기준"""
    per_frame = defaultdict(list)
    n_lost = 0
    for ln in open(path, encoding="utf-8", errors="ignore"):
        v = ln.split()
        if len(v) < 6:
            continue
        try:
            x1, y1, x2, y2, fno = (int(v[i]) for i in (1, 2, 3, 4, 5))
        except ValueError:
            continue
        # 7번째 열이 lost — 화면 밖으로 나간 상자다. 넣으면 빈 자리를 학습한다.
        if len(v) > 6 and v[6] == "1":
            n_lost += 1
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        acts = [a for a in re.findall(r'"([^"]*)"', ln)[1:] if a]
        cls = AMBIG
        for a in acts:                     # 자세 라벨이 있으면 그것을 쓴다
            if a in POSE_TO_CLS:
                cls = POSE_TO_CLS[a]
                break
        per_frame[fno].append((cls, x1, y1, x2, y2))
    return per_frame, n_lost


def main():
    ap = argparse.ArgumentParser(description="Okutama 자세 3클래스 학습셋 생성")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="처리할 영상 id. 생략하면 오프셋이 있는 전부")
    ap.add_argument("--val-videos", nargs="*", default=["2.2.10", "2.1.2"],
                    help="검증용 영상. **fallen 이 든 영상이어야 한다**")
    ap.add_argument("--stride", type=int, default=15,
                    help="N 프레임마다 1장. 30fps 기준 15=0.5초")
    ap.add_argument("--min-px", type=int, default=12,
                    help="1920 기준 이보다 작은 상자는 버린다 (탐지 하한 20px 의 절반)")
    ap.add_argument("--limit", type=int, default=0, help="영상당 최대 장수 (0=제한 없음)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if not OFFSETS_JSON.exists():
        raise SystemExit(f"프레임 오프셋이 없습니다: {OFFSETS_JSON}\n"
                         f"먼저 okutama_align.py 를 실행하세요.")
    offsets = json.load(open(OFFSETS_JSON, encoding="utf-8"))

    vids = []
    for mp4 in sorted(SRC.rglob("*.mp4")):
        vid = mp4.stem
        if args.videos and vid not in args.videos:
            continue
        lab = LAB_DIR / f"{vid}.txt"
        if not lab.exists():
            continue
        off = offsets.get(vid, {}).get("offset")
        if off is None:
            print(f"  건너뜀 {vid} — 프레임 오프셋 미확정")
            continue
        vids.append((vid, mp4, lab, int(off)))

    if not vids:
        raise SystemExit("처리할 영상이 없습니다")

    out = Path(args.out).resolve()      # 상대경로면 ultralytics 가 configs 기준으로 푼다
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    print(f"대상 {len(vids)}편 · 출력 {OUT_W}x{OUT_H} · stride {args.stride}\n")
    stat = {sp: Counter() for sp in ("train", "val")}
    per_vid = {}

    for vid, mp4, lab, offset in vids:
        split = "val" if vid in args.val_videos else "train"
        per_frame, n_lost = parse_labels(lab)
        if not per_frame:
            print(f"  {vid}: 라벨 없음 — 건너뜀")
            continue

        # 뽑을 디코딩 위치. 화면 p 는 라벨 프레임 p+offset 에 대응한다.
        wanted = sorted({f - offset for f in per_frame
                         if (f - offset) >= 0 and (f - offset) % args.stride == 0})
        if args.limit:
            wanted = wanted[:args.limit]
        want = set(wanted)

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"  {vid}: 영상 열기 실패")
            continue

        c = Counter()
        fno = -1
        made = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fno += 1
            if fno not in want:
                continue
            boxes = per_frame.get(fno + offset, [])
            if not boxes:
                continue
            H, W = frame.shape[:2]
            sx, sy = OUT_W / W, OUT_H / H
            small = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            lines = []
            for cls, x1, y1, x2, y2 in boxes:
                bx1, by1 = x1 * sx, y1 * sy
                bx2, by2 = x2 * sx, y2 * sy
                bx1, by1 = max(0.0, bx1), max(0.0, by1)
                bx2, by2 = min(float(OUT_W), bx2), min(float(OUT_H), by2)
                bw, bh = bx2 - bx1, by2 - by1
                if bw <= 0 or bh <= 0 or max(bw, bh) < args.min_px:
                    c["작아서 제외"] += 1
                    continue
                cx, cy = (bx1 + bx2) / 2 / OUT_W, (by1 + by2) / 2 / OUT_H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw/OUT_W:.6f} {bh/OUT_H:.6f}")
                c[cls] += 1
            if not lines:
                continue

            stem = f"oku_{vid.replace('.', '_')}_f{fno:06d}"
            ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok2:
                continue
            buf.tofile(str(out / "images" / split / f"{stem}.jpg"))
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            made += 1
        cap.release()

        per_vid[vid] = (split, made, c)
        stat[split]["images"] += made
        for k in (PERSON, FALLEN, AMBIG):
            stat[split][k] += c[k]
        print(f"  {vid:<8} {split:<5} {made:>5,}장 | person {c[PERSON]:>6,} "
              f"fallen {c[FALLEN]:>5,} ambig {c[AMBIG]:>6,} | offset {offset:>3} "
              f"| 작아서 제외 {c['작아서 제외']:,}")

    yml = BASE_DIR / "configs" / "data_okutama3.yaml"
    yml.write_text(
        "# Okutama 자세 3클래스 — 0 person / 1 fallen / 2 ambiguous\n"
        "# 원본 4K 를 1920x1080 으로 축소만 했다 (자르지도 확대하지도 않음).\n"
        "# 사람 크기가 우리 운용 조건(54deg/1080p/16m, 선자세 59px)과 일치한다.\n"
        f"train: {(out / 'images' / 'train').as_posix()}\n"
        f"val: {(out / 'images' / 'val').as_posix()}\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n", encoding="utf-8")

    print(f"\n=== Okutama 3클래스 ===")
    print(f"{'분할':<7}{'이미지':>8}{'person':>10}{'fallen':>9}{'ambiguous':>12}")
    for sp in ("train", "val"):
        c = stat[sp]
        if not c["images"]:
            continue
        print(f"  {sp:<5}{c['images']:>8,}{c[PERSON]:>10,}{c[FALLEN]:>9,}{c[AMBIG]:>12,}")
    print(f"\n  설정: {yml}\n  출력: {out}")


if __name__ == "__main__":
    main()
