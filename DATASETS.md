# 데이터셋 확보 안내

서버(Ubuntu 22.04)에서 원본 데이터를 받는 절차. 모두 `drone_dev/data/raw/` 아래에 둔다.

> **재배포 금지** — NOMAD·WiSARD·Okutama 는 연구용 라이선스다. 우리 저장소에 올리지 않는다.
> `.gitignore` 에 `data/` 가 들어 있다.

---

## 0. 준비

```bash
cd ~/drone_yolo          # 저장소 루트
mkdir -p data/raw && cd data/raw
sudo apt update && sudo apt install -y git-lfs unzip p7zip-full aria2 tmux
git lfs install
df -h .                  # 여유 공간 확인 — 원본만 200GB 안팎, 산출물 포함 시 더
```

**받는 동안 세션이 끊겨도 되게 `tmux` 안에서 할 것.**

```bash
tmux new -s dl
```

---

## 1. NOMAD — 배우 100명 · 우리 인물 다양성의 핵심

Bernal 외, *NOMAD: A Natural, Occluded, Multi-scale Aerial Dataset for Emergency Response
Scenarios*, WACV 2024. 42,825 프레임 · 5.4K 영상에서 추출 · 가시도 10단계 라벨.

- 논문 <https://arxiv.org/abs/2309.09518>
- 배포 <https://github.com/ArtRuss/NOMAD>

```bash
git clone https://github.com/ArtRuss/NOMAD.git NOMAD_repo
# 저장소 안내에 따라 실제 이미지·라벨을 받아 아래 구조로 맞춘다
```

**우리 스크립트가 기대하는 구조** (`nomad_prep.py`)

```
data/raw/NOMAD/
├── activityLabels.json      # 배우·거리별 활동 구간 (자세 라벨의 근거)
├── annotations.json
├── metadata.json
└── <이미지 폴더>
```

> `activityLabels.json` 에 형식이 깨진 구간이 4개 있다(4,606개 중).
> `make_pose3_dataset.py` 의 `parse_span()` 이 복구하므로 그대로 두면 된다.

---

## 2. WiSARD — 겨울·가을 산지 · 56,000장

Broyles, Hayner, Leung, *WiSARD: A Labeled Visual and Thermal Image Dataset for
Wilderness Search and Rescue*, IROS 2022. 가시광 + 열화상.

- 논문 <https://arxiv.org/abs/2309.04453>
- 프로젝트 <https://sites.google.com/uw.edu/wisard/>

프로젝트 페이지의 안내에 따라 받는다. **우리는 가시광(VIS)만 쓴다** — 열화상은
우리 카메라(IMX415)에 없다.

**기대 구조** (`wisard_prep.py`)

```
data/raw/WiSARD/
├── 200704_Baker_FLIR_IR_1/          # 1월 설경 (겨울)
├── 200910_Carnation_FLIR_IR_1/
├── 210924_FHL_Enterprise_VIS_0403/  # 9월 해안 (가을)
└── ...
```

폴더명이 `날짜_장소_센서_번호` 규약이다. `VIS` 가 가시광, `IR` 이 열화상.

---

## 3. Okutama-Action — **평가 전용** · 우리 고도와 일치

Barekatain 외, CVPR 2017 Workshops. 4K 드론 영상 43편 + 프레임별 자세 라벨.

- 공식 <http://okutama-action.org/>

```bash
# 공식 사이트의 배포 링크를 따른다 (TrainSetVideos / TestSetFrames / Labels)
```

**기대 구조** (`okutama_prep.py`, `okutama3_prep.py`)

```
data/raw/okutama/
├── Drone1/{Morning,Noon}/*.mp4
├── Drone2/{Morning,Noon}/*.mp4
├── TrainSetVideos (2)/Drone{1,2}/{Morning,Noon}/*.mp4   # ← 30편이 여기 있다
└── Labels/MultiActionLabels/3840x2160/*.txt
```

> **주의 두 가지**
> ① 영상이 40편이다. `Drone*/*/` 만 보면 10편만 잡힌다 — 우리가 겪은 함정이다.
> ② 영상과 라벨의 프레임 번호가 **영상마다 0~21 어긋나 있다.**
> `okutama_align.py` 로 오프셋을 먼저 추정하지 않으면 상자가 사람에서 빗나간다.

**이 데이터는 학습에 쓰지 않는다.** 배우가 10명 미만이라 다양성이 없고,
우리 운용 고도(환산 17 m)와 정확히 맞는 유일한 외부 데이터라 **정직한 잣대**로 남긴다.

---

## 4. SARD — 자세 학습의 핵심 · 유일하게 재배포 가능

Sambolek & Ivasic-Kos, *Search and Rescue Image Dataset for Person Detection — SARD*,
IEEE DataPort 2021. **CC BY 4.0.** 1,981장에 사람이 직접 붙인 6클래스 자세 라벨.

- 원본 <https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard>
  (DOI `10.21227/ahxm-k331`)
- 우리가 쓴 재배포본 — Roboflow `rescuedby/sard-peykp-lxuf9` v1

```bash
# Roboflow 계정의 API 키가 필요하다
pip install roboflow
python - <<'PY'
from roboflow import Roboflow
rf = Roboflow(api_key="<YOUR_KEY>")
ds = rf.workspace("rescuedby").project("sard-peykp-lxuf9") \
       .version(1).download("yolov8", location="data/raw/sard2")
PY
```

**기대 구조**

```
data/raw/sard2/search-and-rescue-2/
├── train/{images,labels}/    # 1,386장
├── valid/{images,labels}/    #   396장
├── test/{images,labels}/     #   198장   ← 최종 판정 전용, 학습 금지
└── data.yaml                 # nc: 6
```

클래스 순서 — `['Running','Walking','laying_down','not_defined','seated','stands']`

> 자세 라벨은 **SARD 원본 고유**다. Roboflow 사용자가 붙인 것이 아니다.
> 인용은 Sambolek & Ivasic-Kos 2021 로 한다.

---

## 5. 받은 뒤 — 전처리

```bash
cd ~/drone_yolo

python nomad_prep.py
python wisard_prep.py

# Okutama 는 정렬이 먼저다
python okutama_align.py --videos 2.2.10 1.2.10 2.1.2 2.1.7 2.1.4 2.1.1 \
                                  1.2.6 1.1.7 1.1.4 1.2.8 1.2.2 2.2.2 1.1.5 1.1.8 \
                        --range 60 --samples 12
python okutama3_prep.py --videos 2.2.10 1.2.10 2.1.2 2.1.7 2.1.4 2.1.1 \
                                  1.2.6 1.1.7 1.1.4 1.2.8 1.2.2 2.2.2 1.1.5 1.1.8 \
                        --val-videos 2.2.10 2.1.2 --stride 15

python make_pose3_dataset.py --wisard exclude    # 학습셋 (SARD+NOMAD 3클래스)
python make_configs.py                            # ★ yaml 절대경로를 이 서버 기준으로
```

**`make_configs.py` 를 빼먹으면 안 된다.** 저장소의 yaml 에는 Windows 절대경로가 박혀 있다.

### 확인

```bash
python - <<'PY'
from pathlib import Path
for d in ["det/pose3_sn", "det/okutama3", "det/wisard",
          "det/nomad_actor01_10", "raw/sard2/search-and-rescue-2"]:
    p = Path("data") / d
    print(f"{d:<38}{'있음' if p.exists() else '없음'}")
PY
```

기대값 — `pose3_sn` train 9,008장 / val 1,592장 (person 7,562 · fallen 2,755 · ambiguous 1,731)

---

## 6. 추가 후보 (현재 미사용)

우리 조건(하향 90° + 자세 라벨)에 **둘 다 맞는 공개 데이터는 위 넷이 전부**다.
아래는 한쪽이 어긋나 보류한 것들이며, 목적이 바뀌면 다시 볼 것.

| 데이터셋 | 규모 | 왜 보류했나 | 링크 |
|---|---|---|---|
| ForestPersons (ETRI) | 96,482장 · Standing/Sitting/**Lying** | 카메라 높이 **1.5~2 m** — 지상 시점 | <https://huggingface.co/datasets/etri/ForestPersons> |
| AI-Hub 자율주행드론 | 320시간 4K · **한국** | 각도 45° 예시, 90° 존재 미확인 · 내국인 승인 필요 | <https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=190> |
| UAV-Human | 피험자 119명 | 라벨이 **클립 단위**, 넘어짐 클래스 없음 | <https://github.com/sutdcv/UAV-Human> |
| CPD-UAV | 1,061장 · 하향 · 픽셀 마스크 | 자세 라벨 없음 (위장 탐지용) | <https://doi.org/10.3390/drones10060447> |

> ForestPersons 는 **한국 산림·사계절(눈 포함)** 이라 아깝지만 시점이 다르다.
> 지상 1.5 m 에서 본 사람과 상공 16 m 에서 내려다본 사람은 화면에서 아예 다르게 생겼다.
