# drone_yolo — 자율 정찰 드론 관제 시스템 · 비전 파트

항공뷰에서 **요구조자(쓰러진 사람)를 탐지**하고, 화면 좌표를 **실좌표로 환산**해
지도에 신고하는 부분이다. 담당: 정서인.

---

## 지금 부탁하는 작업

개발 노트북이 **VRAM 4GB** 라 학습을 못 돌린다. 시연 장비(**RTX 4080 SUPER 16GB**)에서
아래 셋을 실행해 주면 된다.

**1. 시연 장비 실측** — ✅ **완료** (2026-08-31, RTX 4080 SUPER)

```bash
python bench_models.py --models yolo11s,yolo11m,yolo11l --imgsz 1280 --runs 50
```

| 모델 | 파라미터 | GFLOPs@1280 | FPS | 요건 대비 | VRAM(추론) |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolo11s | 9.44M | 86.9 | 56.5 | 11배 | 0.32 GB |
| yolo11m | 20.09M | 274.1 | **60.8** | 12배 | 0.54 GB |
| yolo11l | 25.34M | 350.5 | 45.1 | 9배 | 0.69 GB |

**yolo11m 이 yolo11s 보다 빠른 건 오류가 아니다.** 연산량이 3.2배인데도 그렇다.
이 크기의 모델은 4080 에서 너무 작아 **GPU 가 논다** — 실제 시간은 행렬 연산이 아니라
전처리·NMS·파이썬 오버헤드·메모리 전송이 잡아먹는다. 그 고정 비용이 지배적이면
모델을 키워도 총 시간이 거의 안 변하고 클럭 변동만으로 순서가 뒤집힌다.
yolo11l 에서야 연산이 유의미해져 45.1 로 떨어진다.

**→ 속도와 VRAM 은 변별력이 없다. 셋 다 요건의 9~12배다. 정확도로만 고른다.**

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

**5. `yolo11l` 도 할지는 4번 결과를 보고 정한다** — 속도로는 배제할 이유가 없다(45 FPS).

- m 이 s 보다 **뚜렷이 좋으면** → 용량이 도움이 된다는 신호. l 도 해볼 가치가 있다
- m 이 s 와 **비슷하거나 나쁘면** → 용량이 이미 포화. l 은 시간 낭비다.
  학습 데이터가 11,615장·**1클래스**뿐이라 이쪽일 가능성이 크다

**왜 이걸 하나** — 지금 운용 모델은 `yolov8s @ imgsz 960` 인데, 학습 이미지 원본이
**1280×720** 이라 960 으로 줄이면서 정보를 버리고 있다. 그 결과 객체의 **26~32%가
64px 미만**으로 들어간다. imgsz 를 원본인 1280 으로 올리면 이 비율이 **0~12%** 로 떨어진다.
4GB 에서는 1280 학습이 불가능해 못 해본 실험이다.

진행 상황은 다른 셸에서 `python train_status.py --name yolo11s_1280 --watch` 로 볼 수 있다.

---

## ⚠ 먼저 읽을 것 — 모르면 반드시 걸리는 것들

### 1. `wisard_sept` 도메인의 mAP 0.0 은 **정상이다**

평가하면 이렇게 나온다. 고장이 아니다.

```
visdrone_baseline   wisard_sept   289장   mAP50 0.0
all_nomad+wisard    wisard_sept   289장   mAP50 0.0
```

이 289장에는 **사람이 한 명도 없다.** 그래서 mAP 를 낼 수 없고, 대신
**오탐(false positive) 측정용**으로 쓴다. "사람 없는 배경에서 몇 건이나
잘못 신고하는가"를 보는 도메인이다. 판단은 `wisard_jan`(겨울) 과
`nomad_summer`(여름) 로 한다.

### 2. torch 는 **CPU 빌드가 설치되기 쉽다**

`pip install torch` 를 그냥 하면 CPU 판이 깔려서 GPU 를 못 쓴다.
아래가 `True` 인지 반드시 확인할 것.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# True 2.13.0+cu130   ← 이렇게 나와야 한다
```

`False` 거나 버전에 `+cpu` 가 붙으면 지우고 CUDA 인덱스로 다시 깔아야 한다.

### 3. 한글이 깨지면 콘솔 인코딩 문제다

윈도우 콘솔이 cp949 라 한글 출력에서 `UnicodeEncodeError` 가 난다.

```bash
set PYTHONIOENCODING=utf-8        # cmd
$env:PYTHONIOENCODING="utf-8"     # PowerShell (또는 chcp 65001)
```

같은 이유로 **경로에 한글이 있으면 `cv2.imread`/`imwrite` 가 조용히 실패한다.**
이 저장소 코드는 `np.fromfile`+`cv2.imdecode` 로 우회해 두었으니, 새 코드를
쓸 때만 주의하면 된다.

### 4. train/val 은 **배우·비행 단위로 나눠져 있다** — 다시 섞지 말 것

- NOMAD → **배우(Actor) 단위** 분할
- WiSARD → **비행(flight) 단위** 분할

연속 프레임은 서로 거의 같은 그림이라, 프레임 단위로 무작위 분할하면
같은 장면이 train 과 val 양쪽에 들어가 **성능이 실제보다 크게 부풀려진다.**
`data.yaml` 이나 분할 방식을 임의로 바꾸면 기존 수치와 비교가 불가능해진다.

### 5. 비교할 때는 **imgsz 를 맞춰야** 한다

`eval_domain.py --imgsz` 는 평가 해상도다. 960 으로 학습한 모델을 1280 으로
평가하면 학습 때와 조건이 달라져 숫자가 왜곡된다. 위 4번 명령이
세 모델을 `--imgsz 1280` 으로 통일해 재는 이유다.

신뢰도 임계값 **0.15** 는 재현율에 2배 가중한 F2 가 최대가 되는 지점으로
정한 값이다(`metrics/threshold_tuning.csv`). 놓치는 것보다 잘못 신고하는
쪽이 낫다는 판단이며, 해상도를 바꿔도 이 값은 그대로 쓴다.

---

## ⚠ 시뮬레이터를 돌릴 때만 해당

학습·평가만 할 거면 건너뛰어도 된다.

### 종료 순서를 지킬 것 — **언리얼 먼저, SITL 나중**

역순으로 끄면 **언리얼이 응답 없음 상태로 멈춘다.** WSL 정리 시
`arducopter`, `sim_vehicle.py` 외에 **`mavproxy` 도 같이** 종료해야 한다.

### Cosys-AirSim 은 원본 AirSim 과 **함수 반환 순서가 다르다**

```python
# 레거시 AirSim : to_eularian_angles()          → (pitch, roll, yaw)
# Cosys-AirSim  : quaternion_to_euler_angles()  → (roll, pitch, yaw)
```

이름만 바꿔 쓰면 **에러 없이 roll 과 pitch 가 뒤바뀐다.** 실제로 겪은 문제다.

### `settings.json` 에 카메라를 직접 정의하면 언리얼이 죽는다

`Vehicles.<차량>.Cameras` 에 `"0"` 같은 카메라를 정의하면 Cosys-AirSim 이
새 카메라를 NaN 위치에 스폰하다 **크래시**한다. 해상도·화각은 반드시
**루트 레벨 `CameraDefaults.CaptureSettings`** 에서 바꿀 것.

### SITL 연결

- `--no-mavproxy` 일 때 GCS 링크는 **TCP 127.0.0.1:5760** (UDP 14550 아님)
- TCP 5760 은 **단일 클라이언트만** 허용 — 재연결 전 이전 소켓을 닫아야 한다
- EKF/GPS 준비까지 부팅 후 **30~60초** 걸린다. arm 이 바로 안 되는 건 정상이다

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
연구 데이터셋이라 재배포하지 않는다.

> **이번에는 방법 B 로 간다.** 개발 노트북에 NOMAD 원본 이미지가 남아 있지 않아
> (용량 확보하며 삭제) 변환 완료본을 통째로 넘길 수 없다.

### 방법 B — 원본을 받아 직접 변환한다 ← **이 방법을 쓴다**

**원본을 아래 경로에 그대로 풀어 놓아야 한다.** 스크립트가 이 위치를 고정으로 본다.

```
drone_dev/
  data/
    NOMAD/
      annotations.json            ← 필수. 이게 없으면 라벨을 못 만든다
      activityLabels.json
      metadata.json
      <이미지들>                   ← 하위 폴더에 있어도 된다 (rglob 으로 훑는다)
    WiSARD/
      200704_Baker_FLIR_IR_1/     ← 비행(flight) 단위 폴더
        *.jpg 와 같은 이름의 *.txt
      200910_Carnation_FLIR_IR_1/
      ...
```

WiSARD 는 **폴더 이름이 곧 비행 식별자**다. `wisard_prep.py` 가 이 이름으로
train/val 을 나누고 촬영 월(9월/1월)을 판별하므로 **폴더명을 바꾸면 안 된다.**

```bash
python nomad_prep.py       # NOMAD  원본 → data/dataset_nomad, dataset_nomad_a11_20
python wisard_prep.py      # WiSARD 원본 → data/dataset_wisard
```

각 스크립트는 `--limit` 로 일부만 먼저 돌려볼 수 있다. 전체는 수십 분 걸린다.

### 방법 A — 변환 완료본을 받는다 (약 3 GB, 이번에는 해당 없음)

`data/dataset_nomad`, `data/dataset_nomad_a11_20`, `data/dataset_wisard` 세 폴더를
받아 `data/` 아래 그대로 두면 된다. WiSARD 쪽은 원본이 남아 있어 이 방법도 되지만,
NOMAD 가 없으므로 둘을 섞지 말고 **방법 B 로 통일하는 편이 낫다.**

> **주의 — 리샘플링 기준이 예전 값이다.**
> 두 스크립트는 원본을 그대로 쓰지 않고, **운용 조건에서 사람이 보일 크기**에
> 맞춰 리샘플링한 뒤 1280×720 으로 크롭한다. 그 기준이 지금
> **고도 20m · FOV 60° · 1280px (GSD 1.804 cm/px, 사람 약 94px)** 로 박혀 있는데,
> 실제 운용 조건은 **FOV 54° · 1920px** 로 바뀌었다 (예산안 5.5mm 렌즈 기준).
> 재계산이 필요하지만 **이번 실험에서는 건드리지 말 것** — 기존 모델과
> 같은 데이터로 비교해야 imgsz·백본 효과만 분리해서 볼 수 있다.

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

- **데이터셋 리샘플링 기준이 FOV 60° 시절 값이다.** 운용 조건은 54° 로 바뀌었다.
  위 "방법 B" 의 주의 참고. 이번 실험에서는 그대로 두는 게 맞다.
- `run_experiment.py` 가 자체 MAVLink 연결 함수를 써서 `patrol_detect.FLIGHT_PARAMS`
  (경사각 8° 제한) 가 **적용되지 않는다.** 미해결.
- 자세 분류(정상/쓰러짐 2클래스)는 네 차례 시도 후 F1 0.37~0.47 에서 막혀 중단했다.
  현재는 **1클래스 탐지기**로 쓰러진 사람을 *발견*만 한다.
- 비행 루프 FPS(3.3~5.9)의 병목은 **YOLO 가 아니라** 언리얼 렌더링과 1080p 이미지
  RPC 전송이다. YOLO 단독은 같은 4GB 노트북에서 38 FPS 가 나온다.
  **모델을 키워도 루프 FPS 는 거의 안 떨어진다.**
