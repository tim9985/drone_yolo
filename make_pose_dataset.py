"""
make_pose_dataset.py — 자세 2클래스 학습셋 생성 (정상 / 쓰러짐)

왜 필요한가
  현재 탐지기는 `person` 한 클래스만 낸다. 출력이 "사람 있음"에서 끝나므로
  운영자가 영상을 다시 봐야 위급 여부를 안다. 발표자료 슬라이드 11의
  "쓰러짐 탐지"를 실체화하려면 자세를 클래스로 구분해야 한다.

클래스 정의
  0 person  정상 — Walking + Hiding(서 있는 은폐)
  1 fallen  쓰러짐 — Hiding (Laying) + Laying

  은폐(Hiding)를 별도 클래스로 두지 않는 이유: 은폐는 자세가 아니라 '가려짐' 상태다.
  종횡비 분포에서 보행과 겹침이 0.83 으로 형태상 구분이 사실상 불가능하다.
  대신 가림 정도는 클래스가 아니라 속성으로 따로 다룬다.

라벨 근거
  NOMAD `activityLabels.json` 의 프레임 구간별 활동 라벨.
  크롭 파일명 Actor{배우}_a{거리}_f{프레임} 으로 역추적한다.
  **크롭은 100% 단일 인물**이므로 프레임 활동을 상자에 그대로 적용해도 모호성이 없다.

디스크
  이미지는 하드링크로 연결한다(복사 아님). NTFS 에서 추가 용량이 들지 않는다.
  하드링크가 실패하면 복사로 넘어간다.

실행: python make_pose_dataset.py [--out data/pose_archive/det2]
출력: dataset_pose/{images,labels}/{train,val} + configs/data_pose.yaml
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
SRC_SETS = ("det/nomad_actor01_10", "det/nomad_actor11_20", "det/nomad_actor21_30")
ACT_JSON = BASE_DIR / "data" / "raw" / "NOMAD" / "activityLabels.json"
CLASS_NAMES = ["person", "fallen"]      # 0=정상, 1=쓰러짐


def activity_lookup():
    tbl = {int(r["id"]): r["labels"] for r in json.load(open(ACT_JSON, encoding="utf-8"))}

    def f(actor, dist, frame):
        lab = tbl.get(int(actor), {}).get(str(dist))
        if not lab:
            return None
        for act, rngs in lab.items():
            for s, e in rngs:
                if int(s) <= frame <= int(e):
                    return act
        return None
    return f


def link_or_copy(src, dst):
    """하드링크 우선 — 8,822장을 복사하면 1.2GB 가 그냥 늘어난다."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "pose_archive" / "det2"))
    args = ap.parse_args()
    out = Path(args.out)

    if not ACT_JSON.exists():
        raise SystemExit(f"활동 라벨이 없습니다: {ACT_JSON}")
    look = activity_lookup()

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stat = {sp: Counter() for sp in ("train", "val")}
    skipped = Counter()

    for ds in SRC_SETS:
        for sp in ("train", "val"):
            img_dir = BASE_DIR / "data" / ds / "images" / sp
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", img.stem)
                if not m:
                    skipped["파일명 형식 불일치"] += 1
                    continue
                act = look(m.group(1), int(m.group(2)), int(m.group(3)))
                if act is None:
                    skipped["활동 라벨 없음"] += 1
                    continue
                cls = 1 if "Laying" in act else 0

                lab = (img_dir.parent.parent / "labels" / sp / img.name).with_suffix(".txt")
                if not lab.exists():
                    skipped["라벨 파일 없음"] += 1
                    continue
                lines = [l.split() for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
                lines = [v for v in lines if len(v) == 5]
                if not lines:
                    skipped["빈 라벨"] += 1
                    continue

                new = "\n".join(f"{cls} {' '.join(v[1:])}" for v in lines) + "\n"
                (out / "labels" / sp / f"{img.stem}.txt").write_text(new, encoding="utf-8")
                link_or_copy(img, out / "images" / sp / img.name)
                stat[sp][cls] += len(lines)
                stat[sp]["images"] += 1

    yml = BASE_DIR / "configs" / "data_pose.yaml"
    yml.write_text(
        "# 자세 2클래스 — 0 person(정상) / 1 fallen(쓰러짐)\n"
        "# 근거: NOMAD activityLabels.json, 크롭은 100% 단일 인물\n"
        f"train: {(out / 'images' / 'train').as_posix()}\n"
        f"val: {(out / 'images' / 'val').as_posix()}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n", encoding="utf-8")

    print("=== 자세 2클래스 학습셋 ===")
    for sp in ("train", "val"):
        n0, n1 = stat[sp][0], stat[sp][1]
        tot = n0 + n1
        if not tot:
            continue
        print(f"  {sp:5} 이미지 {stat[sp]['images']:>5}장 | "
              f"정상 {n0:>5} ({n0/tot*100:4.1f}%) | 쓰러짐 {n1:>5} ({n1/tot*100:4.1f}%) | "
              f"불균형 {max(n0,n1)/max(min(n0,n1),1):.1f}:1")
    if skipped:
        print("  제외:", dict(skipped))
    print(f"  설정 파일: {yml}")


if __name__ == "__main__":
    main()
