"""
make_pose3_dataset.py — 자세 3클래스 통합 학습셋 생성

왜 새로 만드는가
  기존 make_pose_dataset.py 는 2클래스였고 라벨 규칙이 이랬다.
      cls = 1 if "Laying" in act else 0
  NOMAD 는 자세가 아니라 **활동 구간**만 준다. 위 규칙은 구간 안이면 무조건
  fallen 이라, 눕는 중·일어나는 중인 전환 동작이 전부 fallen 에 섞였다.
  반쯤 구부린 자세를 "누움"으로 배우면 결정 경계가 흐려진다.

  또 하나. SARD 단독 학습(sard2_pose6)은 laying_down 재현율 0.972 를 냈지만
      · 다른 도메인에서 무너졌고 (Okutama 0.088 / NOMAD 0.163)
      · 원래 있던 탐지 강건성을 잃었다 (NOMAD 사람 발견율 0.667 → 0.290)
  한 도메인만 학습하면 다른 도메인이 무너지는 것은 이미 겪은 일이다.
  nomad30 이 WiSARD 를 0.544 → 0.242 로 떨어뜨렸고, 세 데이터를 합친
  stage1_all 이 0.922 로 되살렸다. 같은 처방을 자세에도 쓴다.

클래스
  0 person     확실히 서있음 · 걷기 · 앉음
  1 fallen     확실히 누워 있음
  2 ambiguous  전환 동작 + 판정 불가 → 운용에서 '판단 보류 → 스냅샷 전송'

  전환 구간을 **지우지 않고** ambiguous 로 보내는 이유:
  NOMAD 크롭은 1인 1장이라 라벨만 지우면 "거기 사람이 없다"를 가르치게 된다.
  미탐지를 학습시키는 셈이라 더 나쁘다.

출처별 매핑
  SARD    laying_down                          → fallen
          Walking · stands · seated · Running  → person
          not_defined                          → ambiguous   (제작자가 애매하다고 판정)
  NOMAD   Laying / Hiding (Laying)  전환 제외   → fallen
          Walking · Hiding                     → person
          전환 구간 (경계 ±trim 프레임)         → ambiguous
  WiSARD  자세 라벨 없음                        → --wisard 로 결정

  SARD 의 자세 라벨은 Roboflow 사용자가 붙인 것이 아니라 **원본 고유**다.
  Sambolek & Ivasic-Kos, "Search and Rescue Image Dataset for Person
  Detection - SARD", IEEE DataPort, 2021 (DOI 10.21227/ahxm-k331).

전환 판정
  프레임 f 의 누움 여부와 f±trim 의 누움 여부가 다르면 전환으로 본다.
  구간 경계 양쪽을 대칭으로 보므로 '눕는 중'과 '일어나는 중'이 모두 잡힌다.

실행
  python make_pose3_dataset.py                      # WiSARD → ambiguous (기본)
  python make_pose3_dataset.py --wisard person      # 변형 B
  python make_pose3_dataset.py --trim-frames 30     # 전환 폭 1.0초
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["person", "fallen", "ambiguous"]
PERSON, FALLEN, AMBIG = 0, 1, 2

NOMAD_SETS = ("det/nomad_actor01_10", "det/nomad_actor11_20", "det/nomad_actor21_30")
ACT_JSON = BASE_DIR / "data" / "raw" / "NOMAD" / "activityLabels.json"
WISARD_DIR = BASE_DIR / "data" / "det" / "wisard"
SARD_DIR = BASE_DIR / "data" / "raw" / "sard2" / "search-and-rescue-2"

# SARD 원본 6클래스 → 3클래스
# ['Running','Walking','laying_down','not_defined','seated','stands']
SARD_MAP = {0: PERSON, 1: PERSON, 2: FALLEN, 3: AMBIG, 4: PERSON, 5: PERSON}


# ─────────────────────────────────────────────────────────────
# NOMAD 활동 구간 → 자세
# ─────────────────────────────────────────────────────────────
def parse_span(rn):
    """구간 하나를 (시작, 끝) 으로 읽는다.

    원본 activityLabels.json 4,606개 구간 중 4개가 깨져 있다.
        ['0174.0334']      쉼표 대신 마침표   (actor 13 / dist 90)
        ['1140.1598']      같은 형태          (actor 42 / dist 70)
        ['0876', '1320}']  닫는 중괄호 혼입   (actor 70 / dist 30)
        ['0728']           끝값 자체가 없음   (actor 44 / dist 90)
    앞의 셋은 의도가 분명하므로 복구한다. 마지막은 끝을 알 수 없으니
    한 프레임짜리 구간으로 좁혀 둔다 — 없는 값을 지어내지 않는다.
    """
    nums = []
    for v in (rn if isinstance(rn, list) else [rn]):
        nums += [int(x) for x in re.findall(r"\d+", str(v))]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    return None


def build_nomad_lookup():
    """(actor, dist) → {frame 구간: 누움 여부} 조회기를 만든다."""
    tbl = {int(r["id"]): r["labels"] for r in json.load(open(ACT_JSON, encoding="utf-8"))}

    # (actor, dist) → [(start, end, is_laying), ...]
    spans = defaultdict(list)
    n_fixed = 0
    for actor, per_dist in tbl.items():
        for dist, acts in per_dist.items():
            for act, rngs in acts.items():
                laying = "Laying" in act
                for rn in rngs:
                    se = parse_span(rn)
                    if se is None:
                        continue
                    if len(rn) != 2:
                        n_fixed += 1
                    spans[(actor, int(dist))].append((se[0], se[1], laying))
    for k in spans:
        spans[k].sort()
    if n_fixed:
        print(f"  주의 — 원본 주석의 깨진 구간 {n_fixed}건을 복구했다")

    def is_laying(actor, dist, frame):
        """해당 프레임의 누움 여부. 구간 밖이면 None."""
        for s, e, lay in spans.get((actor, dist), ()):
            if s <= frame <= e:
                return lay
        return None

    def classify(actor, dist, frame, trim):
        cur = is_laying(actor, dist, frame)
        if cur is None:
            return None
        # 앞뒤 trim 프레임과 상태가 다르면 전환 구간으로 본다
        for off in (-trim, trim):
            nb = is_laying(actor, dist, frame + off)
            if nb is not None and nb != cur:
                return AMBIG
        return FALLEN if cur else PERSON

    return classify


def link_or_copy(src, dst):
    """하드링크 우선 — 1만 장을 복사하면 수 GB 가 그냥 늘어난다."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label(lab_path, mapper):
    """라벨 파일을 읽어 클래스만 바꾼 텍스트를 돌려준다. 못 쓰면 None."""
    try:
        raw = lab_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    out = []
    for ln in raw.splitlines():
        v = ln.split()
        if len(v) != 5:
            continue
        c = mapper(int(v[0]))
        if c is None:          # 제외 대상
            continue
        out.append(f"{c} {' '.join(v[1:])}")
    return ("\n".join(out) + "\n") if out else None


def main():
    ap = argparse.ArgumentParser(description="자세 3클래스 통합 학습셋 생성")
    ap.add_argument("--out", default=str(BASE_DIR / "data" / "det" / "pose3_merged"))
    ap.add_argument("--trim-frames", type=int, default=15,
                    help="전환으로 볼 구간 경계 폭(프레임). 30fps 기준 15=0.5초")
    ap.add_argument("--wisard", choices=["ambiguous", "person", "exclude"],
                    default="ambiguous",
                    help="WiSARD 는 자세 라벨이 없다. 어느 클래스로 넣을지 (D-1)")
    ap.add_argument("--sard-nd", choices=["ambiguous", "drop", "person"],
                    default="ambiguous",
                    help="SARD not_defined 처리. 'drop' 은 상자를 지워 배경으로 만든다 — "
                         "이 클래스에 가려짐·과소 크기가 섞여 있어 모델이 "
                         "'흐릿한 덩어리 = ambiguous' 를 배웠고 실영상에서 벤치·가방에 "
                         "상자가 붙었다")
    ap.add_argument("--wisard-val", action="store_true",
                    help="WiSARD 를 검증셋에도 넣는다. 기본은 학습셋 전용 — "
                         "WiSARD 라벨은 자세 정답이 아니므로 검증에 섞으면 "
                         "best.pt 가 '눈밭 = ambiguous' 쪽으로 뽑힌다")
    ap.add_argument("--name", default=None, help="configs 야믈 이름 (기본: 출력 폴더명)")
    args = ap.parse_args()

    # 반드시 절대경로로 굳힌다. 상대경로를 그대로 야믈에 적으면 ultralytics 가
    # **야믈이 있는 configs/ 기준**으로 풀어서 configs/data/det/... 을 찾다 죽는다.
    out = Path(args.out).resolve()
    trim = args.trim_frames

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    stat = {sp: Counter() for sp in ("train", "val")}
    src_stat = defaultdict(Counter)
    skipped = Counter()

    def emit(img, text, sp, prefix, source):
        """이미지 하드링크 + 라벨 기록. 파일명은 출처 접두어로 충돌을 막는다."""
        stem = f"{prefix}_{img.stem}"
        (out / "labels" / sp / f"{stem}.txt").write_text(text, encoding="utf-8")
        link_or_copy(img, out / "images" / sp / f"{stem}{img.suffix}")
        n = len(text.strip().splitlines())
        stat[sp]["images"] += 1
        for ln in text.strip().splitlines():
            stat[sp][int(ln.split()[0])] += 1
            src_stat[source][int(ln.split()[0])] += 1
        src_stat[source]["images"] += 1
        return n

    # ── 1. SARD ───────────────────────────────────────────────
    # 원본 고유 자세 라벨. test 분할은 최종 검증용으로 남긴다.
    # not_defined(원본 클래스 3) 를 어디로 보낼지는 --sard-nd 가 정한다.
    # 'drop' 은 None 이라 상자가 지워진다 = 그 자리를 **배경으로** 학습한다.
    _nd = {"ambiguous": AMBIG, "drop": None, "person": PERSON}[args.sard_nd]
    sard_map = {**SARD_MAP, 3: _nd}

    print(f"[1/3] SARD …  (not_defined → {args.sard_nd})")
    if not SARD_DIR.is_dir():
        print(f"  건너뜀 — 폴더 없음: {SARD_DIR}")
    else:
        for src_sp, dst_sp in (("train", "train"), ("valid", "val")):
            idir = SARD_DIR / src_sp / "images"
            if not idir.is_dir():
                continue
            for img in sorted(list(idir.glob("*.jpg")) + list(idir.glob("*.png"))):
                lab = SARD_DIR / src_sp / "labels" / (img.stem + ".txt")
                if not lab.exists():
                    skipped["SARD 라벨 없음"] += 1
                    continue
                text = remap_label(lab, lambda c: sard_map.get(c))
                if text is None:
                    skipped["SARD 빈 라벨"] += 1
                    continue
                emit(img, text, dst_sp, "sard", "SARD")

    # ── 2. NOMAD ──────────────────────────────────────────────
    # 활동 구간 → 자세. 전환 구간은 ambiguous 로 보낸다.
    print(f"[2/3] NOMAD … (전환 폭 ±{trim}프레임 = {trim/30:.2f}초)")
    if not ACT_JSON.exists():
        print(f"  건너뜀 — 활동 라벨 없음: {ACT_JSON}")
    else:
        classify = build_nomad_lookup()
        for ds in NOMAD_SETS:
            for sp in ("train", "val"):
                idir = BASE_DIR / "data" / ds / "images" / sp
                if not idir.is_dir():
                    continue
                for img in sorted(idir.glob("*.jpg")):
                    m = re.match(r"Actor(\d+)_a(\d+)_f(\d+)", img.stem)
                    if not m:
                        skipped["NOMAD 파일명 불일치"] += 1
                        continue
                    cls = classify(int(m.group(1)), int(m.group(2)), int(m.group(3)), trim)
                    if cls is None:
                        skipped["NOMAD 활동 라벨 없음"] += 1
                        continue
                    lab = (idir.parent.parent / "labels" / sp / img.name).with_suffix(".txt")
                    if not lab.exists():
                        skipped["NOMAD 라벨 없음"] += 1
                        continue
                    text = remap_label(lab, lambda _c, k=cls: k)
                    if text is None:
                        skipped["NOMAD 빈 라벨"] += 1
                        continue
                    emit(img, text, sp, "nomad", "NOMAD")

    # ── 3. WiSARD ─────────────────────────────────────────────
    # 자세 라벨이 없다. 종횡비로 대신하려 했으나 기준이 성립하지 않았다 —
    # NOMAD 진짜 fallen 의 59.4% 가 세로로 길다(ar<=1.0). 하향 시점에서는
    # 누워도 각도에 따라 세로로 보이기 때문이다.
    print(f"[3/3] WiSARD … (→ {args.wisard})")
    if args.wisard == "exclude":
        print("  제외 지정 — 건너뜀")
    elif not WISARD_DIR.is_dir():
        print(f"  건너뜀 — 폴더 없음: {WISARD_DIR}")
    else:
        wcls = AMBIG if args.wisard == "ambiguous" else PERSON
        # WiSARD 라벨은 자세 정답이 아니다. 검증에 섞으면 ambiguous 가 검증셋의
        # 최다 클래스가 되어, val mAP 로 고르는 best.pt 가 '눈밭이면 ambiguous'
        # 쪽으로 뽑힌다. 겨울 도메인 탐지 강건성은 학습에 남기되 평가는
        # eval_domain 으로 따로 본다.
        w_splits = ("train", "val") if args.wisard_val else ("train",)
        if not args.wisard_val:
            print("  검증셋 제외 — 학습에만 사용 (--wisard-val 로 변경)")
        for sp in w_splits:
            idir = WISARD_DIR / "images" / sp
            if not idir.is_dir():
                continue
            for img in sorted(list(idir.glob("*.jpg")) + list(idir.glob("*.png"))):
                lab = WISARD_DIR / "labels" / sp / (img.stem + ".txt")
                if not lab.exists():
                    skipped["WiSARD 라벨 없음"] += 1
                    continue
                text = remap_label(lab, lambda _c, k=wcls: k)
                if text is None:
                    skipped["WiSARD 빈 라벨"] += 1
                    continue
                emit(img, text, sp, "wisard", "WiSARD")

    # ── 설정 파일 ─────────────────────────────────────────────
    name = args.name or out.name
    yml = BASE_DIR / "configs" / f"data_{name}.yaml"
    yml.write_text(
        "# 자세 3클래스 통합 — 0 person / 1 fallen / 2 ambiguous\n"
        f"# WiSARD 배정: {args.wisard} · NOMAD 전환 폭: ±{trim}프레임\n"
        "# SARD 자세 라벨은 원본 고유 (Sambolek & Ivasic-Kos 2021)\n"
        f"train: {(out / 'images' / 'train').as_posix()}\n"
        f"val: {(out / 'images' / 'val').as_posix()}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n", encoding="utf-8")

    # ── 보고 ──────────────────────────────────────────────────
    print("\n=== 자세 3클래스 통합 학습셋 ===")
    print(f"{'분할':<7}{'이미지':>8}{'person':>10}{'fallen':>10}{'ambiguous':>12}{'불균형':>9}")
    for sp in ("train", "val"):
        c = stat[sp]
        tot = c[PERSON] + c[FALLEN] + c[AMBIG]
        if not tot:
            continue
        vals = [c[PERSON], c[FALLEN], c[AMBIG]]
        ratio = max(vals) / max(min(vals), 1)
        print(f"  {sp:<5}{c['images']:>8,}{c[PERSON]:>10,}{c[FALLEN]:>10,}"
              f"{c[AMBIG]:>12,}{ratio:>8.1f}:1")

    print(f"\n{'출처':<9}{'이미지':>8}{'person':>10}{'fallen':>10}{'ambiguous':>12}")
    for s in ("SARD", "NOMAD", "WiSARD"):
        c = src_stat.get(s)
        if not c:
            continue
        print(f"  {s:<7}{c['images']:>8,}{c[PERSON]:>10,}{c[FALLEN]:>10,}{c[AMBIG]:>12,}")

    if skipped:
        print("\n  제외:", dict(skipped))
    print(f"\n  설정 파일: {yml}")
    print(f"  출력: {out}")


if __name__ == "__main__":
    main()
