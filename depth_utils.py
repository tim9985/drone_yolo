"""
depth_utils.py — Depth 배열 시각화 (작업 1)

collect_data.py 가 저장한 depth_XXXX.npy (미터 단위 float 2D)를 사람이 볼 수 있는
컬러맵 PNG로 변환한다. 거리 범위(min/max)를 좌상단에 표기해 스케일을 알 수 있게 한다.

실행:
  python depth_utils.py                      # dataset/ 의 depth_*.npy 전부 변환
  python depth_utils.py --npy dataset/depth_0000.npy --out d0.png
"""
import argparse
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

# AirSim DepthPlanar는 하늘 등 미측정 영역에 매우 큰 값(1e5 이상)을 넣는다.
# 그대로 정규화하면 지면이 전부 같은 색으로 뭉개지므로 상한을 둔다.
FAR_CLIP_M = 200.0


def imwrite_u(path, img):
    """cv2.imwrite는 한글 경로(캡스톤)에서 실패할 수 있어 numpy 경유로 저장."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(path))


def colorize_depth(depth, far_clip=FAR_CLIP_M, colormap=cv2.COLORMAP_JET,
                    annotate=True, label=""):
    """depth(미터 float 2D) → BGR 컬러맵 이미지. 거리 범위를 좌상단에 표기."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0) & (d < far_clip)
    if not valid.any():
        vis = np.zeros((*d.shape, 3), np.uint8)
        if annotate:
            cv2.putText(vis, "no valid depth", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    dmin = float(d[valid].min())
    dmax = float(d[valid].max())

    clipped = np.clip(d, dmin, dmax)
    clipped[~valid] = dmax  # 하늘/무한대는 최원거리로 취급
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(norm, colormap)

    if annotate:
        txt = f"depth {dmin:.1f}~{dmax:.1f}m"
        if label:
            txt = f"{label} {txt}"
        # 저해상도(256x144)에서도 글자가 잘리지 않도록 폭에 비례한 폰트 크기 사용
        scale = max(0.32, min(0.8, vis.shape[1] / 1600.0))
        thick_out = 3 if scale > 0.5 else 2
        y = int(18 * max(scale / 0.5, 0.8))
        for color, thick in (((0, 0, 0), thick_out), ((255, 255, 255), 1)):
            cv2.putText(vis, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)
    return vis


def depth_thumbnail(depth, size=(200, 150), far_clip=FAR_CLIP_M):
    """오버레이용 소형 썸네일 (작업 4에서 재사용)."""
    vis = colorize_depth(depth, far_clip=far_clip, annotate=False)
    return cv2.resize(vis, size, interpolation=cv2.INTER_AREA)


def visualize_npy(npy_path, out_path=None, far_clip=FAR_CLIP_M):
    npy_path = Path(npy_path)
    depth = np.load(npy_path)
    vis = colorize_depth(depth, far_clip=far_clip, label=npy_path.stem)
    out_path = Path(out_path) if out_path else npy_path.with_suffix(".vis.png")
    imwrite_u(out_path, vis)
    valid = np.isfinite(depth) & (depth > 0) & (depth < far_clip)
    rng = (float(depth[valid].min()), float(depth[valid].max())) if valid.any() else (0, 0)
    print(f"{npy_path.name} -> {out_path.name}  ({depth.shape[1]}x{depth.shape[0]}, "
          f"{rng[0]:.1f}~{rng[1]:.1f}m)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", help="단일 파일 변환")
    ap.add_argument("--out", help="출력 경로 (단일 변환 시)")
    ap.add_argument("--dir", default=str(BASE_DIR / "data" / "misc" / "depth_samples"), help="일괄 변환 폴더")
    ap.add_argument("--far-clip", type=float, default=FAR_CLIP_M)
    args = ap.parse_args()

    if args.npy:
        visualize_npy(args.npy, args.out, args.far_clip)
        return

    files = sorted(Path(args.dir).glob("depth_*.npy"))
    if not files:
        raise SystemExit(f"depth_*.npy 없음: {args.dir} (collect_data.py 먼저 실행)")
    for f in files:
        visualize_npy(f, far_clip=args.far_clip)
    print(f"\n{len(files)}개 변환 완료 → {args.dir}")


if __name__ == "__main__":
    main()
