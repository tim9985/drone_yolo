# drone_yolo — 자율 정찰 드론 관제 시스템 · 비전 파트

항공뷰에서 **요구조자(쓰러진 사람)를 탐지**하고, 화면 좌표를 **실좌표로 환산**해
지도에 신고하는 부분이다. 담당: 정서인.

---

## 지금 부탁하는 작업

개발 노트북이 **VRAM 4GB** 라 학습을 못 돌린다. 시연 장비(**RTX 4080 SUPER 16GB**)에서
아래 셋을 실행해 주면 된다.

**1. 시연 장비 실측** (2분) — 가중치를 받지 않고 구조만으로 재므로 바로 된다.

```bash
python bench_models.py --models yolo11s,yolo11m,yolo11l --imgsz 1280 --runs 50
```

**2. 후보 A 학습** (1~2시간)

```bash
python train_person.py --stage 1 --weights yolo11s.pt --imgsz 1280 --batch -1 --name yolo11s_1280
```

**3. 후보 B 학습** (2~4시간)

```bash
python train_person.py --stage 1 --weights yolo11m.pt --imgsz 1280 --batch -1 --name yolo11m_1280
```

**4. 셋을 같은 잣대로 비교** (10분) — 기존 운용 모델까지 넣어 한 번에 잰다.

```bash
python eval_domain.py --imgsz 1280 \
  --weights runs_person/yolo11s_1280/weights/best.pt \
            runs_person/yolo11m_1280/weights/best.pt \
            weights/yolov8s_stage1_all.pt
```

`--batch -1` 은 VRAM 에 맞춰 자동으로 잡으라는 뜻이다. 16GB 면 알아서 크게 잡는다.
`yolo11s.pt` / `yolo11m.pt` 사전학습 가중치는 처음 실행할 때 자동으로 받아온다.

**왜 이걸 하나** — 지금 운용 모델은 `yolov8s @ imgsz 960` 인데, 학습 이미지 원본이
**1280×720** 이라 960 으로 줄이면서 정보를 버리고 있다. 그 결과 객체의 **26~32%가
64px 미만**으로 들어간다. imgsz 를 원본인 1280 으로 올리면 이 비율이 **0~12%** 로 떨어진다.
4GB 에서는 1280 학습이 불가능해 못 해본 실험이다.

진행 상황은 다른 셸에서 `python train_status.py --name yolo11s_1280 --watch` 로 볼 수 있다.

---

## 설치

```bash
conda create -n drone python=3.10.20
conda activate drone

# torch 는 CUDA 빌드라 공식 인덱스에서 받아야 한다
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

`nvidia-smi` 로 GPU 가 잡히는지, `python -c "import torch; print(torch.cuda.is_available())"`
가 `True` 인지 먼저 확인할 것.

---

## 데이터셋 — **저장소에 없다**

`data/` 는 **18.3 GB · 15만 파일**이고, NOMAD·WiSARD 는 각자 라이선스가 있는 공개
연구 데이터셋이라 재배포하지 않는다. 두 가지 방법 중 하나로 준비한다.

### 방법 A — 정서인에게 변환 완료본을 받는다 (권장, 약 3 GB)

`data/dataset_nomad`, `data/dataset_nomad_a11_20`, `data/dataset_wisard` 세 폴더만
받아서 `data/` 아래 그대로 두면 된다. 외장 하드나 네트워크 공유로 전달.

### 방법 B — 원본을 받아 직접 변환한다

원본 NOMAD / WiSARD 를 받은 뒤:

```bash
python nomad_prep.py       # NOMAD  원본 → YOLO 학습셋
python wisard_prep.py      # WiSARD 원본 → YOLO 학습셋
```

### 준비됐는지 확인

```bash
python -c "from pathlib import Path; [print(p, len(list((Path('data')/p/'images'/'train').glob('*')))) for p in ('dataset_nomad','dataset_nomad_a11_20','dataset_wisard')]"
```

기대값: `2657`, `2240`, `6718` (합계 11,615장 · 객체 18,132개 · 클래스 1개)

---

## 현재 성능 (비교 기준선)

운용 중인 `yolov8s_stage1_all` (imgsz 960, NOMAD+WiSARD 학습):

| 모델 | 여름 mAP50 | 겨울 mAP50 | 쓰러진 사람 발견 |
| --- | ---: | ---: | ---: |
| 기성 VisDrone (사전학습만) | 0.133 | 0.544 | 0.048 |
| **통합 (운용 중)** | **0.641** | **0.922** | **0.531** |

**새 후보가 이 숫자를 넘어야 교체할 이유가 된다.** 특히 여름(0.641)이 관건이다 —
겨울보다 낮은 게 현재 가장 큰 약점이고, imgsz 를 올려서 개선을 기대하는 부분이다.

---

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| `train_person.py` | 탐지 모델 학습 |
| `eval_domain.py` | **도메인 분리 평가** — 계절(여름/겨울)·데이터셋별로 나눠 잰다 |
| `bench_models.py` | 모델별 추론 FPS·VRAM 측정 (가중치 없이 구조만으로 잰다) |
| `train_status.py` | 학습 진행 상황 (`--watch` 로 실시간) |
| `patrol_detect.py` | 시뮬레이터 정찰 비행 + 탐지 + 지도 + KML |
| `coord_transform.py` | 픽셀 → 실좌표 변환 (자세 보정 포함) |
| `kml_export.py` | 탐지 결과를 KML 로 — 지도에 바로 올라간다 |
| `tracking.py` | 광학흐름 / ByteTrack / BoT-SORT 추적 |
| `merge_tune.py` | 병합 반경·최소 관측 횟수 조정 |
| `nomad_prep.py`, `wisard_prep.py` | 원본 → YOLO 학습셋 변환 |

시뮬레이터 관련 파일(`patrol_detect.py` 등)은 **Unreal + AirSim + ArduPilot SITL 이
있어야** 돌아간다. 학습·평가만 할 거면 신경 쓰지 않아도 된다.

---

## 운용 조건 (예산안 v2.0 기준)

| 항목 | 값 |
| --- | --- |
| 카메라 | Sony IMX415 · 1920×1080 |
| 화각 | **수평 54.0°** (5.5mm 렌즈 계산값. v2.0 의 "60°" 는 대각선) |
| 장착 | **고정 하향 90°** · 짐벌 없음 |
| 순찰 고도 | 16 m (시뮬레이터) |
| 신뢰도 임계값 | **0.15** (재현율 2배 가중 F2 최대 구간) |
| 추론 위치 | 중앙 서버 |

짐벌이 없어 **기체가 기울면 좌표가 밀린다.** 고도 16m 기준 실측으로
기울기 2°대에서 산포 0.24~0.70 m, 20°대에서 6.8~10.9 m 였다.
자세한 내용은 `../드론_사양_및_예산_기준서.md` 참고.

---

## 알려진 문제

- `run_experiment.py` 가 자체 MAVLink 연결 함수를 써서 `patrol_detect.FLIGHT_PARAMS`
  (경사각 8° 제한) 가 **적용되지 않는다.** 미해결.
- 자세 분류(정상/쓰러짐 2클래스)는 네 차례 시도 후 F1 0.37~0.47 에서 막혀 중단했다.
  현재는 **1클래스 탐지기**로 쓰러진 사람을 *발견*만 한다.
