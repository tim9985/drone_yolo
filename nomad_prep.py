"""
nomad_prep.py — NOMAD 원본 → 운용 조건에 맞춘 YOLO 학습셋 변환

왜 그냥 못 쓰는가
  NOMAD 원본은 5472x3078 이고 사람 크기가 거리별로 27~233px 로 제각각이다.
  우리 운용 조건(고도 20m, FOV 60°, 1280px, GSD 1.804cm/px)에서 사람은 약 94px 이다.
  원본을 그대로 넣으면 추론 해상도로 축소될 때 사람이 10px 이하로 뭉개져 학습이 안 된다.
  → 사람 크기를 목표 픽셀로 맞춰 리샘플링한 뒤 1280x720 창을 크롭한다.

거리별 필요 배율 (실측 중앙값 기준)
  a10 233px → 0.40배(축소, 최상)   a30 78px → 1.2배(최적)
  a50  47px → 2.0배(한계)          a70/a90 → 2.8~3.5배 업스케일이라 기본 제외

과적합 대응
  · train/val 을 **배우(Actor) 단위**로 분할한다. 같은 배우의 연속 프레임이 양쪽에
    섞이면 성능이 부풀려진다(프레임 간 상관이 매우 높음).
  · 목표 사람 크기에 지터를 줘 스케일 다양성을 만든다.
  · 크롭 위치도 지터를 줘 사람이 항상 중앙에 오지 않게 한다.

실행:
  python nomad_prep.py                        # 기본: a10,a30 사용
  python nomad_prep.py --distances 10,30,50 --per-image 2
  python nomad_prep.py --min-visibility 30    # 가림 심한 표본 제외
출력: data/det/nomad_actor01_10/{images,labels}/{train,val} + data.yaml + nomad_prep_stats.json
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
NOMAD_DIR = BASE_DIR / "data" / "raw" / "NOMAD"
OUT_ROOT = BASE_DIR / "data" / "det" / "nomad_actor01_10"

CROP_W, CROP_H = 1280, 720
# 운용 조건(고도 20m, FOV 60°, 1280px, GSD 1.804cm/px)에서 사람 1.7m = 94px
TARGET_PERSON_PX = 94
SCALE_JITTER = (0.75, 1.35)     # 목표 크기에 곱하는 지터 → 스케일 다양성
MAX_UPSCALE = 2.2               # 이 배율을 넘는 확대는 화질이 무너져 버린다
VAL_ACTOR_RATIO = 0.2
SEED = 42


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img, quality=92):
    ext = Path(path).suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def index_images(root):
    """파일명 → 실제 경로"""
    idx = {}
    for p in root.rglob("*.jpg"):
        idx[p.name] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nomad", default=str(NOMAD_DIR))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--distances", default="10,30",
                    help="사용할 거리(m) 목록. 기본 10,30 (업스케일 과한 50/70/90 제외)")
    ap.add_argument("--per-image", type=int, default=1, help="원본 1장당 생성할 크롭 수")
    ap.add_argument("--min-visibility", type=int, default=0,
                    help="이 값 미만 가시성 표본 제외 (0=전부 사용)")
    ap.add_argument("--target-px", type=int, default=TARGET_PERSON_PX)
    ap.add_argument("--limit", type=int, default=0, help="디버그용 최대 처리 수")
    ap.add_argument("--all-train", action="store_true",
                    help="이번 배치를 전부 train 으로 (검증셋을 기존 것으로 고정할 때)")
    args = ap.parse_args()

    random.seed(SEED)
    nomad = Path(args.nomad)
    out_root = Path(args.out)
    dists = {int(d) for d in args.distances.split(",") if d.strip()}

    ann_path = nomad / "annotations.json"
    if not ann_path.exists():
        raise SystemExit(f"주석 없음: {ann_path}")
    records = json.load(open(ann_path, encoding="utf-8"))
    img_index = index_images(nomad)
    print(f"주석 {len(records)}개 / 보유 이미지 {len(img_index)}장")

    # 사용할 레코드 선별
    usable = []
    for r in records:
        p = img_index.get(r["file_name"])
        if p is None or not r["annotations"]:
            continue
        m = re.match(r"Actor(\d+)_a(\d+)_", r["file_name"])
        if not m:
            continue
        actor, dist = m.group(1), int(m.group(2))
        if dist not in dists:
            continue
        boxes = [b for b in r["annotations"]
                 if int(b.get("visibility", 100)) >= args.min_visibility]
        if not boxes:
            continue
        usable.append({"rec": r, "path": p, "actor": actor, "dist": dist, "boxes": boxes})
    if not usable:
        raise SystemExit("조건에 맞는 표본이 없음 — --distances / --min-visibility 확인")

    # ── 배우 단위 train/val 분할 (프레임 단위로 나누면 누수) ──
    actors = sorted({u["actor"] for u in usable})
    if args.all_train:
        # 이미 확정된 검증셋(이전 배치의 val 배우)을 그대로 쓰기 위해 이번 배치는 전부 train.
        # 검증셋을 바꾸면 이전에 학습한 모델들과 수치를 비교할 수 없다.
        val_actors = set()
    else:
        random.shuffle(actors)
        n_val = max(1, round(len(actors) * VAL_ACTOR_RATIO))
        val_actors = set(actors[:n_val])
    print(f"배우 {len(actors)}명 → val {sorted(val_actors) or '없음(전부 train)'} / "
          f"train {len(actors)-len(val_actors)}명")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    stats = {"crops": 0, "skipped_upscale": 0, "skipped_read": 0,
             "by_dist": Counter(), "by_split": Counter(),
             "person_px": [], "visibility": Counter()}

    processed = 0
    for u in usable:
        if args.limit and processed >= args.limit:
            break
        img = imread_u(u["path"])
        if img is None:
            stats["skipped_read"] += 1
            continue
        H, W = img.shape[:2]
        split = "val" if u["actor"] in val_actors else "train"

        for k in range(args.per_image):
            # 기준 박스: 가장 큰 사람
            bx, by, bw, bh = max(u["boxes"], key=lambda b: b["bbox"][2] * b["bbox"][3])["bbox"]
            long_side = max(bw, bh)
            if long_side <= 1:
                continue
            target = args.target_px * random.uniform(*SCALE_JITTER)
            scale = target / long_side
            if scale > MAX_UPSCALE:
                stats["skipped_upscale"] += 1
                continue

            new_w, new_h = int(round(W * scale)), int(round(H * scale))
            if new_w < CROP_W or new_h < CROP_H:
                # 축소가 너무 커서 크롭 창보다 작아지면 창 크기에 맞춰 하한 적용
                scale = max(CROP_W / W, CROP_H / H)
                new_w, new_h = int(round(W * scale)), int(round(H * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

            # 사람 중심을 기준으로 지터를 준 크롭 창
            pcx, pcy = (bx + bw / 2) * scale, (by + bh / 2) * scale
            jx = random.uniform(-0.30, 0.30) * CROP_W
            jy = random.uniform(-0.30, 0.30) * CROP_H
            x0 = int(round(pcx - CROP_W / 2 + jx))
            y0 = int(round(pcy - CROP_H / 2 + jy))
            x0 = max(0, min(x0, new_w - CROP_W))
            y0 = max(0, min(y0, new_h - CROP_H))
            crop = resized[y0:y0 + CROP_H, x0:x0 + CROP_W]
            if crop.shape[0] != CROP_H or crop.shape[1] != CROP_W:
                continue

            # 크롭 창에 들어온 모든 사람 박스를 YOLO 라벨로
            lines = []
            for b in u["boxes"]:
                sx, sy, sw, sh = [v * scale for v in b["bbox"]]
                x1, y1 = sx - x0, sy - y0
                x2, y2 = x1 + sw, y1 + sh
                # 창 경계로 자른다
                cx1, cy1 = max(x1, 0), max(y1, 0)
                cx2, cy2 = min(x2, CROP_W), min(y2, CROP_H)
                if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                    continue
                # 잘려나간 비율이 크면 라벨로 쓰지 않는다(부분 박스는 학습에 해로움)
                if (cx2 - cx1) * (cy2 - cy1) < 0.4 * sw * sh:
                    continue
                w_, h_ = cx2 - cx1, cy2 - cy1
                lines.append(f"0 {(cx1 + w_/2)/CROP_W:.6f} {(cy1 + h_/2)/CROP_H:.6f} "
                             f"{w_/CROP_W:.6f} {h_/CROP_H:.6f}")
                stats["person_px"].append(max(w_, h_))
                stats["visibility"][b.get("visibility", "?")] += 1
            if not lines:
                continue

            stem = f"{u['rec']['image_id'].replace('.jpg','')}_c{k}"
            imwrite_u(out_root / "images" / split / f"{stem}.jpg", crop)
            (out_root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            stats["crops"] += 1
            stats["by_dist"][u["dist"]] += 1
            stats["by_split"][split] += 1

        processed += 1
        if processed % 300 == 0:
            print(f"  {processed}/{len(usable)} 처리 (크롭 {stats['crops']}장)")

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: ['person']\n",
        encoding="utf-8")

    px = np.array(stats["person_px"]) if stats["person_px"] else np.array([0])
    summary = {
        "crops": stats["crops"],
        "train": stats["by_split"]["train"], "val": stats["by_split"]["val"],
        "val_actors": sorted(val_actors),
        "by_distance": dict(stats["by_dist"]),
        "skipped_upscale": stats["skipped_upscale"],
        "person_px_quartiles": np.percentile(px, [0, 25, 50, 75, 100]).round(1).tolist(),
        "target_person_px": args.target_px,
        "crop_size": [CROP_W, CROP_H],
        "visibility": dict(stats["visibility"]),
        "split_policy": "actor-level (프레임 단위 분할 시 누수)",
    }
    (out_root / "nomad_prep_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== 변환 완료 ===")
    print(f"크롭 {summary['crops']}장 (train {summary['train']} / val {summary['val']})")
    print(f"거리별: {summary['by_distance']}  | 업스케일 초과로 제외 {summary['skipped_upscale']}")
    print(f"사람 크기(px) 사분위: {summary['person_px_quartiles']} (목표 {args.target_px})")
    print(f"저장: {out_root}")


if __name__ == "__main__":
    main()
