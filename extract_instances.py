"""
extract_instances.py — 라벨 상자에서 사람만 오려내 '인스턴스 은행'을 만든다

왜 필요한가
  지금 가장 부족한 것은 **쓰러진 사람의 다양성**이다.
      SARD    1,569개  슬로베니아 · 배우 소수
      Okutama 11,922개 일본 공원 · 배우 10명 미만
      WiSARD  **0개**  ← 겨울·눈에 쓰러진 사람 데이터가 하나도 없다
  그런데 시연 환경은 11~12월 산지다. 지금 모델은 눈밭에 쓰러진 사람을
  한 번도 본 적이 없는 채로 겨울 임무에 나가는 셈이다.

  Copy-Paste 증강(Ghiasi 외, CVPR 2021)은 이 빈칸을 직접 메운다.
  SARD·Okutama 의 쓰러진 사람을 WiSARD 설경에 붙이면 없던 조합이 생긴다.

왜 상자를 그대로 붙이면 안 되나
  우리 라벨은 **상자만** 있고 분할 마스크가 없다. 사각형을 그대로 붙이면
  테두리가 남고, 모델은 사람이 아니라 **붙인 자국**을 학습한다.
  방금 겪은 `ambiguous` 물체 오탐과 같은 종류의 실패다.
  그래서 SAM 에 상자를 프롬프트로 줘서 마스크를 뽑는다.

  항공 하향 시점은 사람과 배경(잔디·흙)의 대비가 뚜렷해 SAM 이 잘 맞는 편이다.
  그래도 실패하는 것이 있으므로 아래 품질 검사로 거른다.

품질 검사 — 통과 못 하면 버린다
  · 마스크 면적 / 상자 면적이 0.15~0.85 밖   → 실패(배경째 잡았거나 거의 못 잡음)
  · 마스크가 상자 테두리에 과도하게 붙음      → 배경까지 물었을 가능성
  · 최소 변 길이 미달                        → 너무 작아 붙여도 형태가 안 보임

출력
  runs_instances/<이름>/
    fallen/*.png     RGBA (알파 = 마스크)
    person/*.png
    bank.json        출처·원 크기·마스크 비율 기록
    sheet_*.jpg      눈 검수용 모음판

실행
  python extract_instances.py --source sard --classes fallen --limit 400
  python extract_instances.py --source okutama --classes fallen --limit 300 --device 0
"""
import argparse
import json
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
OUT_ROOT = BASE_DIR / "runs_instances"

# SARD 원본 6클래스 → 우리 이름
SARD_DIR = BASE_DIR / "data" / "raw" / "sard2" / "search-and-rescue-2"
SARD_PICK = {2: "fallen", 1: "person", 5: "person", 4: "person", 0: "person"}
# okutama3 는 이미 3클래스로 만들어 둔 것
OKU_DIR = BASE_DIR / "data" / "det" / "okutama3"
OKU_PICK = {1: "fallen", 0: "person"}
# NOMAD — **배우 100명**. 우리가 가진 유일한 인물 다양성 자산이다.
# SARD·Okutama 는 배우가 8명 남짓이라 중복 제거를 해도 같은 사람만 남는다.
# pose3_sn 에 NOMAD 가 3클래스로 정리돼 있으므로 그대로 쓴다.
NOMAD_DIR = BASE_DIR / "data" / "det" / "pose3_sn"
NOMAD_PICK = {1: "fallen", 0: "person"}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(Path(p).suffix, img)
    if ok:
        buf.tofile(str(p))
    return ok


def gather(source, classes, limit):
    """(이미지경로, 클래스이름, xyxy) 목록"""
    items = []
    if source == "sard":
        for split in ("train", "valid"):
            idir = SARD_DIR / split / "images"
            ldir = SARD_DIR / split / "labels"
            if not idir.is_dir():
                continue
            for ip in sorted(idir.glob("*.jpg")):
                lp = ldir / (ip.stem + ".txt")
                if not lp.exists():
                    continue
                img_wh = None
                for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    v = ln.split()
                    if len(v) < 5:
                        continue
                    nm = SARD_PICK.get(int(v[0]))
                    if nm is None or nm not in classes:
                        continue
                    if img_wh is None:
                        im = imread_u(ip)
                        if im is None:
                            break
                        img_wh = (im.shape[1], im.shape[0])
                    W, H = img_wh
                    x, y, w, h = (float(t) for t in v[1:5])
                    items.append((ip, nm, ((x - w / 2) * W, (y - h / 2) * H,
                                           (x + w / 2) * W, (y + h / 2) * H)))
    elif source == "okutama":
        for split in ("train", "val"):
            idir = OKU_DIR / "images" / split
            ldir = OKU_DIR / "labels" / split
            if not idir.is_dir():
                continue
            for ip in sorted(idir.glob("*.jpg")):
                lp = ldir / (ip.stem + ".txt")
                if not lp.exists():
                    continue
                for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    v = ln.split()
                    if len(v) < 5:
                        continue
                    nm = OKU_PICK.get(int(v[0]))
                    if nm is None or nm not in classes:
                        continue
                    W, H = 1920, 1080          # okutama3 는 고정 해상도
                    x, y, w, h = (float(t) for t in v[1:5])
                    items.append((ip, nm, ((x - w / 2) * W, (y - h / 2) * H,
                                           (x + w / 2) * W, (y + h / 2) * H)))
    elif source == "nomad":
        for split in ("train", "val"):
            idir = NOMAD_DIR / "images" / split
            ldir = NOMAD_DIR / "labels" / split
            if not idir.is_dir():
                continue
            for ip in sorted(idir.glob("nomad_*.jpg")):     # SARD 분은 제외
                lp = ldir / (ip.stem + ".txt")
                if not lp.exists():
                    continue
                img_wh = None
                for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    v = ln.split()
                    if len(v) < 5:
                        continue
                    nm = NOMAD_PICK.get(int(v[0]))
                    if nm is None or nm not in classes:
                        continue
                    if img_wh is None:
                        im = imread_u(ip)
                        if im is None:
                            break
                        img_wh = (im.shape[1], im.shape[0])
                    W, H = img_wh
                    x, y, w, h = (float(t) for t in v[1:5])
                    items.append((ip, nm, ((x - w / 2) * W, (y - h / 2) * H,
                                           (x + w / 2) * W, (y + h / 2) * H)))
    else:
        raise SystemExit(f"모르는 출처: {source}")

    # 고르게 뽑는다. 앞에서 자르면 한 영상·한 배우에 몰린다.
    if limit and len(items) > limit:
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return items


def quality_ok(mask, bw, bh, min_side, ratio_lo, ratio_hi):
    """마스크가 쓸 만한지. (통과여부, 사유)"""
    area = float(mask.sum())
    if area <= 0:
        return False, "빈 마스크"
    r = area / max(bw * bh, 1)
    if not (ratio_lo <= r <= ratio_hi):
        return False, f"면적비 {r:.2f}"
    ys, xs = np.nonzero(mask)
    mh, mw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if min(mh, mw) < min_side:
        return False, f"너무작음 {mw}x{mh}"
    # 테두리에 과하게 붙으면 배경까지 물었을 가능성이 크다
    edge = (mask[0, :].sum() + mask[-1, :].sum()
            + mask[:, 0].sum() + mask[:, -1].sum())
    if edge > 0.35 * (2 * mask.shape[0] + 2 * mask.shape[1]):
        return False, "테두리과다"

    # 종횡비 — 누운 사람도 6:1 을 넘지 않는다. 넘으면 여러 명이 붙었거나 번진 것
    ar = max(mw, mh) / max(min(mw, mh), 1)
    if ar > 6.0:
        return False, f"과도한비율 {ar:.1f}"

    # 볼록성 — 사람은 제법 꽉 찬 덩어리다. 구멍이 숭숭하면 풀·그림자를 물은 것
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "윤곽없음"
    big = max(cnts, key=cv2.contourArea)
    # 조각이 여러 개로 흩어져 있으면 실패 — 가장 큰 조각이 대부분을 차지해야 한다
    if cv2.contourArea(big) < 0.70 * area:
        return False, "조각분산"
    hull = cv2.convexHull(big)
    ha = cv2.contourArea(hull)
    if ha > 0 and cv2.contourArea(big) / ha < 0.45:
        return False, "볼록성부족"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description="라벨 상자 → 사람 인스턴스 은행")
    ap.add_argument("--source", required=True, choices=["sard", "okutama", "nomad"])
    ap.add_argument("--classes", nargs="+", default=["fallen"],
                    choices=["fallen", "person"])
    ap.add_argument("--limit", type=int, default=400, help="추출 시도 개수")
    ap.add_argument("--sam", default="mobile_sam.pt",
                    help="mobile_sam.pt(40MB, 가벼움) 또는 sam_b.pt(375MB, 더 정확)")
    ap.add_argument("--device", default="cpu",
                    help="학습이 GPU 를 쓰는 중이면 cpu 로 둔다")
    ap.add_argument("--pad", type=float, default=0.12,
                    help="상자 주변 여유. SAM 이 경계를 잡을 단서가 필요하다")
    ap.add_argument("--min-side", type=int, default=14)
    ap.add_argument("--min-box-px", type=int, default=60,
                    help="원본 상자 장변 하한. SAM 은 작은 대상에서 약해 "
                         "38~54px 구간에 실패가 몰린다. 어차피 붙일 때 크기를 "
                         "맞추므로 큰 원본을 쓰는 편이 낫다(축소가 확대보다 낫다)")
    ap.add_argument("--ratio-lo", type=float, default=0.15)
    ap.add_argument("--ratio-hi", type=float, default=0.85)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.source}_{'_'.join(args.classes)}"
    out = OUT_ROOT / name
    for c in args.classes:
        (out / c).mkdir(parents=True, exist_ok=True)

    items = gather(args.source, set(args.classes), args.limit)
    if not items:
        raise SystemExit("추출할 상자가 없습니다")
    print(f"출처 {args.source} · 대상 {len(items):,}개 · SAM {args.sam} ({args.device})\n")

    from ultralytics import SAM
    sam = SAM(args.sam)

    bank = []
    rej = Counter()
    cur_path, cur_img = None, None

    for n, (ip, nm, (x1, y1, x2, y2)) in enumerate(items, 1):
        if ip != cur_path:
            cur_img = imread_u(ip)
            cur_path = ip
        if cur_img is None:
            rej["이미지 못 읽음"] += 1
            continue
        H, W = cur_img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if max(bw, bh) < args.min_box_px:
            rej["원본작음"] += 1
            continue
        # 여유를 준 창을 잘라 SAM 에 넣는다. 원본 전체를 넣으면 느리고,
        # 여유가 없으면 SAM 이 경계를 판단할 단서가 부족하다.
        m = args.pad * max(bw, bh)
        cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
        cx2, cy2 = int(min(W, x2 + m)), int(min(H, y2 + m))
        win = cur_img[cy1:cy2, cx1:cx2]
        if win.size == 0 or min(win.shape[:2]) < 8:
            rej["창이 너무 작음"] += 1
            continue

        box = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
        try:
            r = sam(win, bboxes=[box], verbose=False, device=args.device)[0]
        except Exception as e:
            rej[f"SAM 오류"] += 1
            continue
        if r.masks is None or len(r.masks.data) == 0:
            rej["마스크 없음"] += 1
            continue
        mask = r.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != win.shape[:2]:
            mask = cv2.resize(mask, (win.shape[1], win.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        ok, why = quality_ok(mask, bw, bh, args.min_side, args.ratio_lo, args.ratio_hi)
        if not ok:
            rej[why.split()[0]] += 1
            continue

        ys, xs = np.nonzero(mask)
        my1, my2, mx1, mx2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = win[my1:my2, mx1:mx2]
        a = mask[my1:my2, mx1:mx2] * 255
        # 가장자리를 살짝 부드럽게 — 딱 잘린 테두리는 붙인 티가 난다
        a = cv2.GaussianBlur(a, (3, 3), 0)
        rgba = np.dstack([rgb, a])

        stem = f"{nm}_{n:05d}"
        imwrite_u(out / nm / f"{stem}.png", rgba)
        bank.append(dict(file=f"{nm}/{stem}.png", cls=nm, source=args.source,
                         src_image=ip.name,
                         box_px=round(float(max(bw, bh)), 1),
                         mask_w=int(mx2 - mx1), mask_h=int(my2 - my1),
                         fill=round(float(mask.sum()) / max(bw * bh, 1), 3)))
        if n % 100 == 0:
            print(f"  {n}/{len(items)} · 채택 {len(bank)} · 탈락 {sum(rej.values())}")

    (out / "bank.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== 인스턴스 은행 ===")
    per = Counter(b["cls"] for b in bank)
    for c in args.classes:
        print(f"  {c:<8}{per.get(c,0):>6,}개")
    print(f"  채택률 {len(bank)}/{len(items)} = {len(bank)/len(items)*100:.1f}%")
    if rej:
        print(f"  탈락 사유: {dict(rej)}")
    if bank:
        px = [b["box_px"] for b in bank]
        print(f"  원 상자 크기 중앙 {int(np.median(px))}px · 범위 {int(min(px))}~{int(max(px))}px")
    # ── 검수판 — 체커보드 위에 얹어 마스크 경계를 눈으로 본다 ──
    # 흰 배경에 얹으면 잘린 팔다리가 안 보인다. 격자 무늬라야 알파 구멍이 드러난다.
    if bank:
        for cls in args.classes:
            picks = [b for b in bank if b["cls"] == cls][:60]
            if not picks:
                continue
            S, COLS = 128, 10
            rows = (len(picks) + COLS - 1) // COLS
            sheet = np.full((rows * (S + 22) + 30, COLS * S, 3), 30, np.uint8)
            cv2.putText(sheet, f"{cls.upper()} · {len(picks)}/{sum(1 for b in bank if b['cls']==cls)}"
                        f" · 격자가 비쳐야 정상 (마스크 경계 확인)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            for i, b in enumerate(picks):
                im = cv2.imdecode(np.fromfile(str(out / b["file"]), np.uint8),
                                  cv2.IMREAD_UNCHANGED)
                if im is None or im.shape[2] != 4:
                    continue
                h, w = im.shape[:2]
                s = min(S / w, S / h)
                im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
                # 체커보드
                tile = np.zeros((S, S, 3), np.uint8)
                q = 8
                yy, xx = np.mgrid[0:S, 0:S]
                tile[((yy // q + xx // q) % 2) == 0] = 70
                tile[((yy // q + xx // q) % 2) == 1] = 110
                ih, iw = im.shape[:2]
                oy, ox = (S - ih) // 2, (S - iw) // 2
                a = im[:, :, 3:4].astype(np.float32) / 255.0
                roi = tile[oy:oy + ih, ox:ox + iw].astype(np.float32)
                tile[oy:oy + ih, ox:ox + iw] = (im[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
                r, c = divmod(i, COLS)
                y0 = 30 + r * (S + 22)
                sheet[y0:y0 + S, c * S:(c + 1) * S] = tile
                cv2.putText(sheet, f"{b['box_px']:.0f}px f{b['fill']:.2f}",
                            (c * S + 3, y0 + S + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (200, 200, 200), 1, cv2.LINE_AA)
            imwrite_u(out / f"sheet_{cls}.jpg", sheet)
            print(f"  검수판: sheet_{cls}.jpg")

    print(f"\n  출력: {out}")
    print("  ※ 붙이기 전에 sheet 로 눈 검수할 것 — 마스크가 어긋난 것을 넣으면"
          " 모델이 '붙인 자국'을 학습한다")


if __name__ == "__main__":
    main()
