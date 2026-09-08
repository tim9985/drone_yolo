"""
dedup_instances.py — 인스턴스 은행에서 사실상 같은 것을 걸러낸다

왜 필요한가
  Copy-Paste 의 목적은 **다양성**이다. 그런데 영상에서 뽑은 인스턴스는
  연속 프레임이라 거의 같은 것이 수십 개씩 들어온다.
  실제로 Okutama 417개를 눈으로 보니 같은 인물(청바지·올리브 상의)의
  거의 같은 자세가 25번 넘게 반복됐다. 이걸 그대로 붙이면 다양성이 느는 게 아니라
  **한 사람의 겉모습이 증폭**된다 — 애초에 피하려던 문제를 더 키우는 셈이다.

어떻게 거르나
  모양과 색을 합친 작은 지문을 만들어 비교한다.
    · 알파 마스크를 16x16 으로 줄인 것        → 자세·윤곽
    · RGB 를 8x8 로 줄여 마스크 안만 평균낸 것 → 옷·피부 색
  지문 사이 거리가 임계값 아래면 같은 것으로 보고 하나만 남긴다.

  회전은 무시한다. 붙일 때 어차피 임의로 돌리므로, 같은 사람이 같은 자세로
  누운 것은 각도가 달라도 같은 것으로 취급하는 편이 낫다.
  그래서 지문에 **회전 8방향의 최소 거리**를 쓴다.

실행
  python dedup_instances.py --bank runs_instances/oku_fallen --thresh 0.14
  python dedup_instances.py --bank runs_instances/sard_fallen --thresh 0.12 --apply
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def imwrite_u(p, img):
    ok, buf = cv2.imencode(Path(p).suffix, img)
    if ok:
        buf.tofile(str(p))
    return ok


def fingerprint(rgba):
    """(모양 16x16, 색 8x8x3) 지문. 값은 0~1."""
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    shape = cv2.resize(a, (16, 16), interpolation=cv2.INTER_AREA)
    n = shape.sum()
    if n > 0:
        shape = shape / max(shape.max(), 1e-6)

    rgb = rgba[:, :, :3].astype(np.float32)
    # 마스크 밖은 색 평균에 넣지 않는다. 배경색이 지문을 흐린다.
    m3 = np.dstack([a] * 3)
    num = cv2.resize(rgb * m3, (8, 8), interpolation=cv2.INTER_AREA)
    den = cv2.resize(m3, (8, 8), interpolation=cv2.INTER_AREA)
    color = num / np.maximum(den, 1e-3) / 255.0
    return shape, color


def rot_variants(shape):
    """회전 8방향 — 붙일 때 임의로 돌리므로 각도 차이는 같은 것으로 본다."""
    out = []
    s = shape
    for _ in range(4):
        out.append(s)
        out.append(np.fliplr(s))
        s = np.rot90(s)
    return out


def dist(fa, fb):
    sa, ca = fa
    sb, cb = fb
    ds = min(np.abs(v - sb).mean() for v in rot_variants(sa))
    dc = np.abs(ca - cb).mean()
    return 0.55 * ds + 0.45 * dc


def main():
    ap = argparse.ArgumentParser(description="인스턴스 은행 중복 제거")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--thresh", type=float, default=0.13,
                    help="이 거리보다 가까우면 같은 것으로 본다. 크게 줄수록 많이 걸러진다")
    ap.add_argument("--apply", action="store_true",
                    help="실제로 bank.json 을 갱신하고 중복 파일을 _dup/ 로 옮긴다")
    args = ap.parse_args()

    bank_dir = Path(args.bank)
    meta = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    print(f"{bank_dir.name} — 인스턴스 {len(meta):,}개 · 임계값 {args.thresh}")

    fps = []
    for m in meta:
        im = imread_u(bank_dir / m["file"])
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            fps.append(None)
            continue
        fps.append(fingerprint(im))

    keep, drop = [], []
    kept_fp = []
    for i, m in enumerate(meta):
        if fps[i] is None:
            drop.append((m, "읽기실패"))
            continue
        dup_of = None
        for j, kf in enumerate(kept_fp):
            if dist(fps[i], kf) < args.thresh:
                dup_of = keep[j]["file"]
                break
        if dup_of is None:
            keep.append(m)
            kept_fp.append(fps[i])
        else:
            drop.append((m, dup_of))

    print(f"  남김 {len(keep):,}개 · 중복 {len(drop):,}개 "
          f"({len(keep)/max(len(meta),1)*100:.1f}% 유지)")

    # 출처별로 얼마나 남았는지 — 한 영상에 몰려 있으면 여전히 문제다
    from collections import Counter
    src = Counter(m.get("src_image", "?") for m in keep)
    print(f"  서로 다른 원본 이미지 {len(src):,}개 · 최다 원본 {src.most_common(1)[0][1]}개")

    if not args.apply:
        print("\n  (미리보기만 — 실제로 적용하려면 --apply)")
        return

    dup_dir = bank_dir / "_dup"
    dup_dir.mkdir(exist_ok=True)
    for m, _why in drop:
        src_p = bank_dir / m["file"]
        if src_p.exists():
            dst = dup_dir / Path(m["file"]).name
            shutil.move(str(src_p), str(dst))
    (bank_dir / "bank.json").write_text(
        json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  적용 완료 · 중복 {len(drop):,}개를 {dup_dir} 로 옮김")
    print(f"  bank.json 갱신 — 이제 {len(keep):,}개")


if __name__ == "__main__":
    main()
