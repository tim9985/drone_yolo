"""
make_configs.py — 데이터 설정(yaml)을 현재 머신 기준으로 다시 만든다

왜 필요한가
  configs/*.yaml 에 **절대 경로가 박혀 있다.** 개발 노트북에서 만들어진
  `C:/Users/timjj/Desktop/캡스톤/drone_dev/...` 가 그대로 들어 있어서,
  다른 컴퓨터나 리눅스 서버에서 clone 하면 학습이 바로 실패한다.

  경로를 상대 경로로 바꾸는 방법도 있지만, ultralytics 가 yaml 위치가 아니라
  실행 위치를 기준으로 푸는 경우가 있어 절대 경로가 더 안전하다.
  그래서 **경로를 고정하는 대신, 이 스크립트로 매번 생성**한다.

  서버에서 처음 할 일:
      git clone ... && cd drone_yolo
      python make_configs.py          # ← 여기서 경로가 그 머신 기준으로 다시 써진다

검증셋 구성은 바꾸지 않는다
  기존 모델(여름 0.641 / 겨울 0.922)과 같은 잣대로 비교해야 하므로
  train/val 분할은 그대로 유지한다. 경로만 바뀐다.

실행:
  python make_configs.py              # 있는 데이터셋만 설정에 넣는다
  python make_configs.py --check      # 생성하지 않고 경로 존재 여부만 확인
"""
import argparse
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
CFG = BASE_DIR / "configs"

# 이름 → (train 하위경로들, val 하위경로들)
NOMAD = ["det/nomad_actor01_10", "det/nomad_actor11_20", "det/nomad_actor21_30"]
WISARD = "det/wisard"

SETS = {
    # 운용 모델이 쓴 구성. NOMAD 배우 1~30 + WiSARD 9월·1월
    "data_all.yaml": {
        "note": ["NOMAD(배우1~30) + WiSARD(9월·1월) 통합",
                 "val: NOMAD 배우 004·008·014·018 + WiSARD 9월 비행 2개 + 1월 시간분할 뒤 30%"],
        "train": [f"{d}/images/train" for d in NOMAD] + [f"{WISARD}/images/train"],
        "val":   [f"{d}/images/val" for d in NOMAD[:2]] + [f"{WISARD}/images/val"],
    },
    # 배우 수 효과를 볼 때 쓴 것들. 검증셋을 고정해 같은 잣대로 비교한다
    "data_nomad20.yaml": {
        "note": ["NOMAD 배우 1~20 (10명 기준선과 비교용). 검증셋 고정"],
        "train": [f"{d}/images/train" for d in NOMAD[:2]],
        "val":   [f"{d}/images/val" for d in NOMAD[:2]],
    },
    "data_nomad30.yaml": {
        "note": ["NOMAD 배우 1~30. 검증셋은 10/20명 실험과 동일하게 고정"],
        "train": [f"{d}/images/train" for d in NOMAD],
        "val":   [f"{d}/images/val" for d in NOMAD[:2]],
    },
}


def build(name, spec, missing):
    lines = [f"# {n}" for n in spec["note"]]
    lines.append("# 이 파일은 make_configs.py 가 생성한다. 직접 고치지 말 것.")
    for key in ("train", "val"):
        paths = []
        for rel in spec[key]:
            p = DATA / rel
            if p.is_dir():
                paths.append(p.as_posix())
            else:
                missing.append(f"{name}:{key} → data/{rel}")
        if not paths:
            return None
        lines.append(f"{key}:")
        lines += [f"  - {p}" for p in paths]
    lines += ["nc: 1", "names: ['person']", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="데이터 설정 yaml 을 현재 머신 기준으로 생성")
    ap.add_argument("--check", action="store_true", help="생성하지 않고 확인만")
    args = ap.parse_args()

    print(f"기준 경로: {BASE_DIR}")
    if not DATA.is_dir():
        raise SystemExit(f"data/ 가 없습니다: {DATA}\n"
                         "README 의 '데이터셋' 절을 따라 먼저 준비하세요.")

    missing, made = [], []
    CFG.mkdir(exist_ok=True)
    for name, spec in SETS.items():
        text = build(name, spec, missing)
        if text is None:
            print(f"  건너뜀  {name}  (필요한 데이터가 하나도 없음)")
            continue
        if not args.check:
            (CFG / name).write_text(text, encoding="utf-8")
        made.append(name)
        print(f"  {'확인' if args.check else '생성'}  configs/{name}")

    if missing:
        print(f"\n없는 경로 {len(missing)}건 — 해당 항목은 설정에서 빠졌다")
        for m in missing:
            print(f"    {m}")

    # 학습에 실제로 몇 장이 들어가는지 세어 둔다. 수치가 예상과 다르면 데이터가 덜 준비된 것이다
    print("\n데이터 규모")
    for rel in NOMAD + [WISARD]:
        tr = DATA / rel / "images" / "train"
        va = DATA / rel / "images" / "val"
        if tr.is_dir():
            print(f"  {rel:<24} train {len(list(tr.glob('*.jpg'))):>5}장 · "
                  f"val {len(list(va.glob('*.jpg'))) if va.is_dir() else 0:>5}장")
        else:
            print(f"  {rel:<24} 없음")
    if not args.check and made:
        print(f"\n완료. 이제 학습할 수 있다:")
        print(f"  python train_person.py --stage 1 --data configs/data_all.yaml "
              f"--imgsz 1280 --batch -1 --name <실험이름>")


if __name__ == "__main__":
    main()
