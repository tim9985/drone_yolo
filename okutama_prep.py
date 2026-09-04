"""
okutama_prep.py — Okutama-Action 4K 영상 → 자세 라벨 YOLO 학습셋

왜 필요한가
  NOMAD 의 자세 라벨은 Walking / Hiding / Laying 3분법이라 **앉은 사람이 어디로
  가는지 모호**했다. Okutama 는 Standing / Sitting / Lying / Walking / Running 을
  따로 라벨링해서, "앉아 있는 사람은 요구조자가 아니다" 를 학습시킬 수 있다.

  그리고 해상도가 결정적이다. 배포된 1280x720 추출 프레임은 사람이 **중앙 39px**
  밖에 안 된다 — 자세 분류가 실패했던 97px 보다도 작다. 반면 원본 4K 영상은
  같은 장면에서 **중앙 118px** 이다. 그래서 mp4 에서 직접 뽑는다.

라벨 형식 (공백 구분)
  track_id  xmin ymin xmax ymax  frame  lost occluded generated  "Person" "자세" "행동"...
  좌표는 **3840x2160 기준**이다. 1280x720 프레임을 쓸 때만 3으로 나눈다.

자세 → 클래스
  Lying              → fallen   (요구조자)
  Standing / Walking → person   (정상)
  Sitting / Running  → person   (정상. 앉은 사람은 구조 대상이 아니다)
  자세 라벨 없음      → 제외     (판단 근거가 없다)

리샘플링
  NOMAD 와 **같은 규약**을 쓴다 — 사람 크기를 목표 픽셀에 맞춘 뒤 창을 잘라낸다.
  그래야 두 데이터셋을 한 학습에 섞을 수 있다. 목표 크기를 바꾸면
  nomad_prep.py 도 같은 값으로 다시 돌려야 한다.

실행:
  python okutama_prep.py --target-px 200          # 실기체 고도 16m 수준
  python okutama_prep.py --target-px 200 --limit 200   # 먼저 조금만
출력:
  data/det/okutama_pose/{images,labels}/{train,val} + data.yaml
"""
import argparse
import json
import random
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
OUT = BASE_DIR / "data" / "det" / "okutama_pose"
OFFSETS_JSON = BASE_DIR / "configs" / "okutama_offsets.json"

LABEL_W, LABEL_H = 3840, 2160          # 라벨 좌표계 = 4K
CROP_W, CROP_H = 1280, 720             # 잘라낼 창 (NOMAD 산출물과 동일)
POSE_SET = {"Standing", "Sitting", "Lying", "Walking", "Running"}
# 0 = person(정상), 1 = fallen(요구조자)
POSE_TO_CLS = {"Lying": 1, "Standing": 0, "Walking": 0, "Sitting": 0, "Running": 0}
CLASS_NAMES = ["person", "fallen"]

# 검증용으로 뺄 영상. **영상 단위로 나눈다** — 같은 영상의 연속 프레임이
# train/val 양쪽에 들어가면 성능이 크게 부풀려진다 (NOMAD 는 배우 단위, WiSARD 는 비행 단위)
#
# **검증셋에 fallen 이 반드시 들어가야 한다.** 처음에 1.1.9 / 2.2.3 을 골랐더니
# 두 영상 다 누운 사람이 0명이라, 정작 재려는 쓰러짐 성능을 측정할 수 없었다.
# Lying 이 있는 영상은 1.1.8(43) · 1.2.10(319) · 2.2.10(341) 셋뿐이다.
# 그중 2.2.10 을 검증으로 뺀다 — 학습에는 1.2.10 과 1.1.8 이 남아
# 양쪽 모두 fallen 을 갖는다. 1.1.9 는 person 다양성 확보용으로 함께 뺀다.
VAL_VIDEOS = {"2.2.10", "1.1.9"}


def imwrite_u(path, img):
    """한글 경로에서 cv2.imwrite 가 조용히 실패한다 — 인코딩 후 직접 쓴다"""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def parse_labels(path):
    """프레임번호 → [(cls, x1, y1, x2, y2)]  (4K 좌표계)"""
    per_frame = defaultdict(list)
    skipped = Counter()
    for ln in open(path, encoding="utf-8", errors="ignore"):
        p = ln.split()
        if len(p) < 10:
            continue
        try:
            x1, y1, x2, y2 = (int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            frame, lost = int(p[5]), int(p[6])
        except ValueError:
            skipped["형식 오류"] += 1
            continue
        if lost:                       # 화면 밖으로 나간 상태
            skipped["lost"] += 1
            continue
        quoted = re.findall(r'"([^"]*)"', ln)
        pose = next((q for q in quoted[1:] if q in POSE_SET), None)
        if pose is None:
            skipped["자세 라벨 없음"] += 1
            continue
        if x2 <= x1 or y2 <= y1:
            skipped["빈 박스"] += 1
            continue
        per_frame[frame].append((POSE_TO_CLS[pose], x1, y1, x2, y2))
    return per_frame, skipped


def label_max_frame(path):
    mx = -1
    for ln in open(path, encoding="utf-8", errors="ignore"):
        q = ln.split()
        if len(q) >= 6:
            try:
                mx = max(mx, int(q[5]))
            except ValueError:
                pass
    return mx


def load_offsets():
    """okutama_align.py 가 추정한 영상별 프레임 오프셋을 읽는다.

    배포된 영상 일부는 **앞부분이 잘려** 라벨 프레임 번호와 어긋나 있다.
    그대로 쓰면 드론이 움직이는 구간마다 상자가 사람 옆 빈 땅에 앉는다.
    실측(탐지기로 정렬 점수 측정):
        1.2.10  오프셋 +21  점수 0.255 → 0.590
        2.2.10  오프셋 +18  점수 0.274 → 0.583
    이 두 영상에 Lying 이 1,640 + 1,739 개 들어 있어서, 살리느냐가
    자세 학습의 성패를 가른다 (정상 6개를 다 합쳐도 211 개뿐).

    offset 이 null 이면 어떤 값으로도 정렬되지 않은 영상이라 제외한다."""
    if not OFFSETS_JSON.exists():
        return None
    try:
        return json.loads(OFFSETS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_videos(strict=True):
    """(영상id, mp4, 라벨, 오프셋) 목록.

    오프셋 파일이 있으면 그것을 따르고, 없으면 예전 방식(프레임 수 비교)으로
    어긋난 영상을 통째로 뺀다. 오프셋 파일을 만들려면 okutama_align.py 를 돌린다."""
    lab_dir = SRC / "Labels" / "MultiActionLabels" / "3840x2160"
    offsets = load_offsets()
    out, dropped = [], []
    for mp4 in sorted(SRC.glob("Drone*/*/*.mp4")):
        vid = mp4.stem
        lab = lab_dir / f"{vid}.txt"
        if not lab.exists():
            continue

        if offsets is not None and vid in offsets:
            off = offsets[vid].get("offset")
            if off is None:
                dropped.append(f"{vid}(정렬 실패)")
                continue
            out.append((vid, mp4, lab, int(off)))
            continue

        # 오프셋 정보가 없는 영상은 보수적으로 처리한다
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        mx = label_max_frame(lab)
        if strict and mx >= n:
            dropped.append(f"{vid}(라벨 {mx} ≥ 영상 {n}, 오프셋 미측정)")
            continue
        out.append((vid, mp4, lab, 0))

    if offsets is None:
        print("  (오프셋 파일 없음 — okutama_align.py 를 먼저 돌리면 손상 영상도 살릴 수 있다)")
    if dropped:
        print(f"  제외 {len(dropped)}개: {', '.join(dropped)}")
    used = [f"{v}{o:+d}" for v, _, _, o in out if o]
    if used:
        print(f"  오프셋 적용: {', '.join(used)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Okutama 4K → 자세 2클래스 YOLO 학습셋")
    ap.add_argument("--target-px", type=int, default=200,
                    help="사람 긴 변을 이 픽셀로 맞춘다. 실기체 고도 16m 가 약 200px")
    ap.add_argument("--stride", type=int, default=10,
                    help="N프레임마다 1장. 연속 프레임은 거의 같은 그림이라 다 쓸 이유가 없다")
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="목표 크기에 주는 무작위 폭. 스케일 다양성을 만든다")
    ap.add_argument("--limit", type=int, default=0, help="영상당 최대 생성 장수 (0=제한 없음)")
    ap.add_argument("--min-px", type=int, default=24,
                    help="이보다 작게 찍힌 사람이 있는 창은 버린다")
    ap.add_argument("--no-strict", action="store_true",
                    help="프레임이 어긋난 영상도 포함한다 (권장하지 않음)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    random.seed(42)
    out_root = Path(args.out)
    vids = find_videos(strict=not args.no_strict)
    if not vids:
        raise SystemExit(f"영상을 찾지 못했습니다: {SRC}/Drone*/*/*.mp4")
    print(f"영상 {len(vids)}개 · 목표 사람 크기 {args.target_px}px · {args.stride}프레임마다\n")

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out_root / sub / split).mkdir(parents=True, exist_ok=True)

    stat = Counter()
    skip_all = Counter()
    for vid, mp4, lab, offset in vids:
        split = "val" if vid in VAL_VIDEOS else "train"
        per_frame, skipped = parse_labels(lab)
        skip_all.update(skipped)
        if not per_frame:
            print(f"  {vid}: 라벨 없음 — 건너뜀")
            continue

        # **순차 디코딩**으로 읽는다. cap.set(CAP_PROP_POS_FRAMES) 로 건너뛰면
        # H.264 에서 정확한 프레임이 아니라 근처 키프레임이 돌아온다 — 라벨과
        # 다른 프레임이 짝지어져 상자가 사람 옆으로 밀린다. 실제로 겪은 버그다.
        # 디코딩 위치 p 의 화면은 라벨 프레임 p+offset 에 해당한다.
        # 그래서 "뽑을 디코딩 위치" 를 라벨 프레임에서 offset 만큼 되돌려 계산한다.
        cap = cv2.VideoCapture(str(mp4))
        wanted = sorted(f - offset for f in per_frame
                        if (f - offset) >= 0 and (f - offset) % args.stride == 0)
        if args.limit:
            wanted = wanted[:args.limit]
        wanted_set = set(wanted)
        last = max(wanted) if wanted else -1
        made = 0
        fno = -1
        while fno < last:
            ok, frame = cap.read()
            if not ok or frame is None:
                stat["프레임 읽기 실패"] += 1
                break
            fno += 1
            if fno not in wanted_set:
                continue
            boxes = per_frame.get(fno + offset, [])
            if not boxes:
                continue

            # 사람 크기를 목표 픽셀에 맞추는 배율. 그 프레임의 중앙값 기준
            sides = [max(x2 - x1, y2 - y1) for _, x1, y1, x2, y2 in boxes]
            med = sorted(sides)[len(sides) // 2]
            if med <= 0:
                continue
            scale = (args.target_px / med) * random.uniform(1 - args.jitter, 1 + args.jitter)

            resized = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            H, W = resized.shape[:2]
            if W < CROP_W or H < CROP_H:
                stat["창보다 작음"] += 1
                continue

            # 창의 중심으로 삼을 사람을 고른다.
            #
            # 무작위로 고르면 안 된다. 한 프레임에 서 있는 사람 10명과 누운 사람
            # 1명이 있으면 10/11 확률로 서 있는 사람을 중심으로 잘라서, 정작
            # 필요한 누운 사람이 창 밖으로 밀려난다. 실제로 그렇게 했더니
            # 이론상 718개 나와야 할 fallen 이 334개만 생성됐다.
            # **희소한 쪽(fallen)을 우선한다.**
            fallen_boxes = [b for b in boxes if b[0] == 1]
            cls0, bx1, by1, bx2, by2 = random.choice(fallen_boxes or boxes)
            cx, cy = (bx1 + bx2) / 2 * scale, (by1 + by2) / 2 * scale
            ox = int(cx - CROP_W / 2 + random.uniform(-0.2, 0.2) * CROP_W)
            oy = int(cy - CROP_H / 2 + random.uniform(-0.2, 0.2) * CROP_H)
            ox = max(0, min(ox, W - CROP_W))
            oy = max(0, min(oy, H - CROP_H))
            crop = resized[oy:oy + CROP_H, ox:ox + CROP_W]

            lines, too_small = [], False
            for cls, x1, y1, x2, y2 in boxes:
                sx1, sy1 = x1 * scale - ox, y1 * scale - oy
                sx2, sy2 = x2 * scale - ox, y2 * scale - oy
                # 창 안에 들어온 부분만 남긴다
                sx1, sy1 = max(0.0, sx1), max(0.0, sy1)
                sx2, sy2 = min(float(CROP_W), sx2), min(float(CROP_H), sy2)
                if sx2 - sx1 < 4 or sy2 - sy1 < 4:
                    continue
                if max(sx2 - sx1, sy2 - sy1) < args.min_px:
                    too_small = True
                    break
                lines.append(f"{cls} {(sx1+sx2)/2/CROP_W:.6f} {(sy1+sy2)/2/CROP_H:.6f} "
                             f"{(sx2-sx1)/CROP_W:.6f} {(sy2-sy1)/CROP_H:.6f}")
                stat[f"cls{cls}"] += 1
            if too_small or not lines:
                stat["창 버림"] += 1
                continue

            stem = f"{vid}_f{fno + offset:05d}"
            imwrite_u(out_root / "images" / split / f"{stem}.jpg", crop)
            (out_root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            stat[f"{split} 장수"] += 1
            made += 1
        cap.release()
        print(f"  {vid:<8} [{split}] {made:>5}장 생성")

    (out_root / "data.yaml").write_text(
        f"# Okutama-Action 4K → 자세 2클래스 (목표 사람 {args.target_px}px)\n"
        f"path: {out_root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n", encoding="utf-8")

    tot = stat["cls0"] + stat["cls1"]
    print(f"\n=== 결과 ===")
    print(f"  train {stat['train 장수']:,}장 · val {stat['val 장수']:,}장")
    if tot:
        print(f"  person(정상) {stat['cls0']:,} ({stat['cls0']/tot*100:.1f}%) · "
              f"fallen(요구조자) {stat['cls1']:,} ({stat['cls1']/tot*100:.1f}%)")
    if skip_all:
        print(f"  라벨 제외: {dict(skip_all)}")
    for k in ("프레임 읽기 실패", "창보다 작음", "창 버림"):
        if stat[k]:
            print(f"  {k}: {stat[k]:,}")
    print(f"  저장: {out_root}")


if __name__ == "__main__":
    main()
