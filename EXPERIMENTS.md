# 서버 학습 실험 계획

대여 서버(**Ubuntu 22.04 · i9-11900 · RTX 3090 24GB · RAM 64GB**)에서 파인튜닝을
다시 돌린다. 목표는 **지금 운용 모델을 이기는 것 하나**다.

## 넘어야 할 기준선

현재 운용 중인 `yolov8s_stage1_all` (imgsz 960 · NOMAD 배우1~30 + WiSARD)

| 지표 | 값 |
| --- | ---: |
| 여름 mAP50 (nomad_summer) | **0.641** |
| 겨울 mAP50 (wisard_jan) | **0.922** |
| 쓰러진 사람 발견 | 0.531 |

> **이 가중치는 안전자산이다.** 서버에서 무엇을 하든 `weights/yolov8s_stage1_all.pt`
> 는 건드리지 않는다. 새 후보가 위 숫자를 넘지 못하면 그냥 안 쓰면 된다.

**여름(0.641)이 관건이다.** 겨울보다 0.28 낮고, 이번 실험의 성패는 여기서 갈린다.

---

## 0. 서버 준비 (약 30분)

```bash
git clone https://github.com/tim9985/drone_yolo.git
cd drone_yolo                       # ← 경로에 한글이 없어야 한다

conda create -n drone python=3.10 -y && conda activate drone
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#  True NVIDIA GeForce RTX 3090   ← 이렇게 나와야 한다
```

데이터를 `data/` 아래 놓은 뒤 **반드시** 설정을 다시 만든다.

```bash
python make_configs.py
```

`configs/*.yaml` 에는 개발 노트북의 절대 경로가 박혀 있다. 이 명령이 그것을
현재 머신 기준으로 다시 쓴다. **건너뛰면 학습이 바로 실패한다.**

### 3090 에 맞춘 설정

노트북(4GB) 기준값이 코드에 남아 있다. 서버에서는 이렇게 바꿔 쓴다.

| 항목 | 노트북 | 서버 | 이유 |
| --- | --- | --- | --- |
| `--batch` | 8 (자동) | **`-1`** | 24GB 에 맞춰 자동으로 크게 잡힌다 |
| `workers` | 4 | **8** | i9-11900 은 8코어 |
| `cache` | `False` | **`True`** | RAM 64GB. 데이터가 다 올라가면 에폭당 시간이 크게 준다 |

`cache` 와 `workers` 는 `train_person.py` 의 `model.train(...)` 호출부에 있다.

---

## 1. 환경 검증 — 제일 먼저 (5분)

**새 환경이 기존 수치를 재현하는지 확인하지 않으면, 이후 비교가 전부 무의미하다.**

```bash
python eval_domain.py --imgsz 960 --weights weights/yolov8s_stage1_all.pt
```

기대값: 여름 **0.641** · 겨울 **0.922**. 이 값이 안 나오면 데이터가 다르거나
설정이 어긋난 것이다. **여기서 멈추고 원인을 찾는다.**

---

## 2. 실험 순서

한 번에 하나씩만 바꾼다. 그래야 무엇이 효과를 냈는지 알 수 있다.
3090 기준 예상 시간은 60에폭 · 11,615장 기준이다.

### 1차 — 해상도와 백본 (데이터 없이 지금 가능, 약 1시간)

| # | 실험 | 명령 | 예상 |
| --- | --- | --- | ---: |
| **E1** | 백본만 교체 | `--weights yolo11s.pt --imgsz 960 --name e1_11s_960` | ~10분 |
| **E2** | 해상도만 상향 | `--weights yolov8s.pt --imgsz 1280 --name e2_v8s_1280` | ~15분 |
| **E3** | 둘 다 | `--weights yolo11s.pt --imgsz 1280 --name e3_11s_1280` | ~15분 |

```bash
python train_person.py --stage 1 --data configs/data_all.yaml \
  --weights yolo11s.pt --imgsz 1280 --batch -1 --name e3_11s_1280
```

**E2 가 핵심이다.** 학습 이미지 원본이 1280×720 인데 imgsz 960 으로 줄여 써 왔다.
그 결과 객체의 **26~32% 가 64px 미만**으로 들어간다. 1280 이면 **0~12%** 로 떨어진다.
4GB 에서는 불가능했던 실험이다.

평가는 **학습 imgsz 와 맞춰서** 한다.

```bash
python eval_domain.py --imgsz 1280 \
  --weights runs_person/e2_v8s_1280/weights/best.pt \
            runs_person/e3_11s_1280/weights/best.pt
```

### 2차 — 데이터 (NOMAD 원본 도착 후, 약 2시간)

원본을 받으면 `metrics/nomad_actor_selection.csv` 의 **42명**으로 변환한다.
연령 5구간 균등 · 동양계 12명 · 상의 색 분산으로 고른 목록이다.

| # | 실험 | 바꾸는 것 | 예상 |
| --- | --- | --- | ---: |
| **E4** | 배우 확대 | 30명 → **42명** (약 10,500장) | ~20분 |
| **E5** | 요구조자 가중 | 누움 구간을 더 자주 샘플링 (20.7% → **40%**) | ~20분 |
| **E6** | E4 + E5 | 둘 다 | ~25분 |

E5 가 필요한 이유 — 지금 학습 데이터의 활동 분포다.

| 활동 | 비중 |
| --- | ---: |
| Walking | **53.8%** |
| Hiding | 25.5% |
| Hiding (Laying) | 14.9% |
| Laying | 5.8% |

**우리 과제는 쓰러져 못 움직이는 사람을 찾는 것인데, 그 표본이 20.7% 뿐이다.**
배우를 늘려도 이 비율은 그대로다 — 변환 단계에서 바꿔야 한다.

> 걷는 사람을 완전히 빼면 안 된다. 순찰 중 멀쩡히 걷는 사람도 탐지돼야
> 오탐/미탐 구분이 되고, '사람'이라는 개념 자체가 좁아진다. **절반 정도**가 적당하다.

### 3차 — 용량 (승자 조합으로, 약 1시간)

| # | 실험 | 조건 |
| --- | --- | --- |
| **E7** | `yolo11m` | 1~2차 승자 조합 위에서 |
| **E8** | `yolo11l` | **E7 이 E3 보다 뚜렷이 좋을 때만** |

판단 기준을 미리 정해 둔다.

- **m 이 s 보다 뚜렷이 좋으면** → 용량이 도움이 된다는 신호. l 도 볼 가치가 있다
- **비슷하거나 나쁘면** → 용량은 이미 포화. l 은 시간 낭비다

후자일 가능성이 크다. 학습 데이터가 **11,615장 · 1클래스**뿐이고, 공개 벤치마크의
성능 차이는 COCO 80클래스를 감당하려고 만든 용량에서 나온다.

속도는 변별력이 없다. RTX 4080 SUPER 실측에서 셋 다 요건(5 FPS)의 9~12배였고,
**yolo11m 이 yolo11s 보다 오히려 빨랐다**(60.8 대 56.5) — 이 크기 모델은 GPU 가 놀고
전처리·NMS 오버헤드가 시간을 지배한다.

---

## 3. 최종 정리 (약 30분)

승자가 정해지면 세 가지를 다시 잰다.

```bash
# 1) 임계값 재산정 — 모델이 바뀌면 운용점도 바뀐다
python tune_threshold.py --weights runs_person/<승자>/weights/best.pt

# 2) 크기 곡선 — 학습 imgsz 가 바뀌면 절벽 위치가 옮겨간다
python eval_scale.py --weights runs_person/<승자>/weights/best.pt

# 3) 촬영 거리별
python eval_altitude.py --imgsz 1280 --weights runs_person/<승자>/weights/best.pt
```

2번을 반드시 해야 한다. 현재 모델은 사람이 **36px(환산 고도 88m)** 까지 성능이
유지되고 24px 에서 무너지는데, **1280 으로 학습하면 그 경계가 이동한다.**
운용 고도 기준(20~40m)이 여기에 달려 있다.

가져올 것 — `best.pt`, `runs_person/<이름>/results.csv`, `metrics/*.csv`

---

## 하지 말 것

- **imgsz 를 1280 보다 크게 올리지 말 것.** 원본이 1280×720 이라 그 이상은
  없는 정보를 늘리는 것뿐이고 시간만 든다. (원본에서 1920×1080 창으로 다시
  변환한다면 이야기가 달라진다 — 그건 별도 실험이다)
- **train/val 분할을 바꾸지 말 것.** NOMAD 는 배우 단위, WiSARD 는 비행 단위로
  나뉘어 있다. 프레임 단위로 다시 섞으면 같은 장면이 양쪽에 들어가 **성능이
  실제보다 크게 부풀려진다.**
- **`weights/yolov8s_stage1_all.pt` 를 덮어쓰지 말 것.** 안전자산이다.
- **초반 에폭 지표로 중단하지 말 것.** `warmup_epochs 3` 때문에 2~4에폭에서
  떨어지는 것이 정상이다. 실제로 2에폭 0.551 → 3에폭 0.310 까지 떨어졌다가
  0.606 으로 회복한 적이 있다.

## 알아 둘 것

- `wisard_sept` 도메인의 mAP 0.0 은 **정상**이다. 사람이 0명인 289장이고
  오탐 측정 전용이다.
- 서버는 대여 중이다. 실험을 순서대로 돌리고, **각 실험이 끝날 때마다
  `results.csv` 와 `best.pt` 를 즉시 내려받아 둔다.**
