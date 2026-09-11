# 가중치 계보

서버에 넘겨받은 `.pt` 파일이 **어디서 왔고, 무엇으로 학습됐고, 어디까지 믿을 수 있는지** 정리한다.
가중치는 `.gitignore` 대상이라 저장소에 없고, 노트북에서 직접 복사해 넘긴다.

작성 2026-09-11 · 근거 `runs_person/*/args.yaml` · `logs/train_*.log` · 학습 당시 실행 명령 기록

---

## 1. 한 장 요약

```
[공개] YOLOv8s-VisDrone (11클래스)          HuggingFace dronefreak/visdrone-yolov8s · best.pt
  │                                          → weights/yolov8s_visdrone.pt   우리가 학습하지 않음
  │
  ├── + NOMAD 배우 1~10                      → stage1_nomad       보관
  ├── + NOMAD 배우 1~20                      → stage1_nomad20     보관
  ├── + NOMAD 배우 1~30                      → stage1_nomad30     보관 (겨울에서 붕괴)
  │
  └── + NOMAD 배우 1~30 + WiSARD             → stage1_all  ★      1클래스 탐지 · 모든 자세 실험의 시작점
        │
        ├── + SARD 6클래스 (동결 없음)          → sard2_pose6        7차 · 탐지 강건성 상실
        ├── + SARD+NOMAD 3클래스 · 백본 동결    → pose3_sn_freeze ★  8차 · 현재 운용
        └── + 위에서 SARD not_defined 제외      → pose3_nd_freeze    9차 · 기각
```

**중요** — NOMAD 단계별 모델과 `stage1_all` 은 **서로 이어 학습한 것이 아니다.**
넷 모두 VisDrone 가중치에서 **각각 따로** 출발했다(실행 시 `--weights` 미지정 →
`train_person.py` 의 기본값 `BASE_WEIGHTS = weights/yolov8s_visdrone.pt`).

---

## 2. 넘긴 파일

| 파일 | 크기 | 필수 | 역할 |
|---|---:|:---:|---|
| `yolov8s_stage1_all.pt` | 64.0 MB | ● | 1클래스 탐지 최고 모델. 모든 실험의 시작 가중치 |
| `yolov8s_pose3_sn_freeze.pt` | 21.5 MB | ● | 3클래스 운용 모델. 서버 기준선 재현용 |
| `yolov8s_pose3_nd_freeze.pt` | 21.5 MB | | 9차. 비교용 |
| `yolov8s_sard2_pose6.pt` | 21.5 MB | | 7차. 망각 대조군 |

`stage1_all` 만 64 MB 인 이유 — 옵티마이저·EMA 상태가 제거되지 않은 채 복사됐다.
**시작 가중치(`--weights`)로 쓰는 데는 문제없다.**

---

## 3. `yolov8s_stage1_all.pt` — 1클래스 탐지

### 출처

| 항목 | 값 |
|---|---|
| 시작 가중치 | `yolov8s_visdrone.pt` (공개 · 11클래스) → 헤드를 1클래스로 재구성 |
| 클래스 | 1개 — `person` |
| 학습 데이터 | NOMAD 배우 1~30 **7,624장** + WiSARD **6,718장** = **14,342장** |
| 검증 데이터 | NOMAD 1,198장 + WiSARD 1,571장 = 2,769장 |
| 설정 | imgsz 960 · batch 8 · lr0 0.01 · patience 15 · 34 epoch (best ep30) |
| 증강 | degrees 10 · **flipud 0** · hsv_s 0.8 · mosaic 1.0 · scale 0.5 |
| 실행 명령 | `python train_person.py --stage 1 --data configs/data_all.yaml` |
| 학습 장비 | 노트북 RTX 3050 4GB · 2026-08-03 |

### 데이터 분할

| 데이터 | 분할 기준 | 비고 |
|---|---|---|
| NOMAD | **배우 단위** — 검증 배우 004·008·014·018 | 같은 농장. 장소는 분리되지 않음 |
| WiSARD 9월 | **비행 단위** — `210924_FHL_Enterprise_VIS_0403`, `_0409` 검증 | |
| WiSARD 1월 | **한 비행을 시간순으로 잘라** 앞 train / 뒤 val | ⚠ 1월 비행이 하나뿐이라 |

### 성능 (학습에 쓰지 않은 검증셋, `metrics/eval_domain.csv`)

| 도메인 | mAP50 | 재현율 |
|---|---:|---:|
| NOMAD 여름 | 0.641 | 0.606 |
| **WiSARD 1월 겨울** | **0.922** | 0.824 |
| 통합 2,769장 | 0.850 | 0.767 |

| 대상 (conf 0.15) | 재현율 |
|---|---:|
| 쓰러진 사람 | 0.608 |
| 부분 가림 | 0.548 (conf 0.25) |
| 배경 오탐 | 0.99건/장 |

### ⚠ 이 수치를 쓸 때 주의

1. **겨울 0.922 는 독립 검증이 아니다.** 같은 비행의 뒷부분으로 쟀다.
   요구명세서 NFR-V03 의 "학습·평가 영상과 장소를 분리한다" 조건을 만족하지 않는다.
2. **하향 시점에 맞지 않는 증강으로 학습됐다.** `degrees 10 · flipud 0` —
   하향 90°에서는 방향이 무의미해 `180 · 0.5` 가 맞다. `REBOOT_PLAN.md` 실험 A 가 이걸 고친다.
3. **`--resume` 으로 쓰지 말 것.** 파일 안에 노트북의 Windows 경로(`C:\Users\...`)가
   학습 인자로 박혀 있어 이어학습이 깨진다. **항상 `--weights` 로 넘겨 새 실행을 만든다.**

---

## 4. `yolov8s_pose3_sn_freeze.pt` — 3클래스 자세 (운용)

### 출처

| 항목 | 값 |
|---|---|
| 시작 가중치 | `yolov8s_stage1_all.pt` |
| 클래스 | 3개 — `person` · `fallen` · `ambiguous` |
| 학습 데이터 | `data/det/pose3_sn` — SARD + NOMAD, **WiSARD 제외** · train 9,008장 / val 1,592장 |
| 라벨 구성 | person 7,562 · fallen 2,755 · ambiguous 1,731 |
| 설정 | imgsz 960 · batch 6 · **lr0 0.001** · **freeze 10(백본 전체)** · 28 epoch (best ep18) |
| 생성 명령 | `python make_pose3_dataset.py --wisard exclude` |
| 학습 명령 | `python train_person.py --stage 1 --data configs/data_pose3_sn.yaml --weights weights/yolov8s_stage1_all.pt --name pose3_sn_freeze --epochs 35 --imgsz 960 --batch 6 --lr0 0.001 --patience 10 --freeze 10` |
| 학습 장비 | 노트북 RTX 3050 4GB · 2026-09-05 · 2시간 33분 |

### 클래스 정의

| 클래스 | 출처 라벨 |
|---|---|
| `person` | SARD `Walking·stands·seated·Running` · NOMAD `Walking·Hiding` |
| `fallen` | SARD `laying_down` · NOMAD `Laying / Hiding (Laying)` 중 **전환 구간 제외** |
| `ambiguous` | SARD `not_defined` · NOMAD 활동 경계 **±15프레임(0.5초)** — 눕는 중·일어나는 중 |

**왜 WiSARD 를 뺐나** — WiSARD 의 `person` 은 "사람이 있다"는 **탐지** 라벨이고,
우리 `person` 은 "서 있다"는 **자세** 라벨이다. 뜻이 다르다. 겨울 강건성은 데이터가 아니라
`stage1_all` 백본을 **동결**해서 지킨다.

### 성능 (`metrics/eval_pose3.csv`, 서버 6단계 기준선)

| 지표 | 값 |
|---|---:|
| **SARD test 쓰러짐 재현율** | **0.974** (정밀도 0.919) |
| NOMAD 사람 발견율 | 0.589 |
| WiSARD 사람 발견율 | 0.804 |
| Okutama(미학습) 사람 발견율 | 0.682 |
| NOMAD 쓰러짐 재현율 | 0.227 |
| Okutama 쓰러짐 재현율 | 0.091 |
| 추론 속도 (1280×720) | 20.6 ms · 48.6 fps |

**`stage1_all` 대비 탐지 손실** — NOMAD −11% · WiSARD −10%.
동결 없이 학습한 7차(`sard2_pose6`)는 −39% · −30% 였다. 동결이 망각을 1/3 로 줄였다.

### 알려진 약점

- **도메인이 바뀌면 쓰러짐 판별이 무너진다** (SARD 0.974 → Okutama 0.091)
- **`ambiguous` 가 물체에 오탐한다** — 실영상에서 벤치·가방에 붙어 전체 탐지의 47%
- 탐지 강건성이 `stage1_all` 보다 약 10% 낮다

---

## 5. 비교용

### `yolov8s_pose3_nd_freeze.pt` — 9차 (기각)

`pose3_sn_freeze` 와 같되 SARD `not_defined` 887개를 **배경으로** 돌렸다.

| 지표 | 8차 | 9차 |
|---|---:|---:|
| SARD test 쓰러짐 | 0.974 | **0.987** |
| 실영상 `ambiguous` 탐지 | 2,012건 | **839건 (−58%)** |
| NOMAD 발견율 | **0.589** | 0.503 |

오탐은 줄었지만 **사람을 더 놓쳐서** 채택하지 않았다. 원인 진단(물체 오탐 = `not_defined`)은 맞았다.

### `yolov8s_sard2_pose6.pt` — 7차

`stage1_all` 에서 **동결 없이** SARD 6클래스만 학습. SARD 쓰러짐 0.960 이지만
NOMAD 발견율 0.403 · WiSARD 0.624 로 **탐지 강건성을 잃었다.** 동결의 필요성을 보여주는 대조군.

클래스 순서 — `['Running','Walking','laying_down','not_defined','seated','stands']`

---

## 6. 처음부터 재현하려면

`stage1_all` 을 서버에서 새로 만들 경우.

```bash
# 1) 시작 가중치 — 공개 VisDrone 모델
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('dronefreak/visdrone-yolov8s', 'best.pt')
shutil.copy(p, 'weights/yolov8s_visdrone.pt')"

# 2) 데이터 (DATASETS.md) → 전처리
python nomad_prep.py && python wisard_prep.py && python make_configs.py

# 3) 학습 — 노트북과 같은 조건
python train_person.py --stage 1 --data configs/data_all.yaml --imgsz 960 --batch 8
```

> 저장소 id 가 비슷한 `dronefreak/yolov8s-visdrone` 은 존재하지 않는다(401).
> `dronefreak/visdrone-yolov8s` 가 맞다.

---

## 7. 서버에서 받은 뒤 확인

```bash
python - <<'PY'
from ultralytics import YOLO
for f in ["yolov8s_stage1_all.pt", "yolov8s_pose3_sn_freeze.pt"]:
    m = YOLO(f"weights/{f}")
    print(f"{f:<30} 클래스 {len(m.names)}개 {list(m.names.values())}")
PY
```

기대 출력

```
yolov8s_stage1_all.pt          클래스 1개 ['person']
yolov8s_pose3_sn_freeze.pt     클래스 3개 ['person', 'fallen', 'ambiguous']
```

그다음 `REBOOT_PLAN.md` 6단계로 기준선 재현 — `pose3_sn_freeze` 수치가 위 4절 표와
±0.01 이내로 맞아야 서버 환경을 믿을 수 있다.
