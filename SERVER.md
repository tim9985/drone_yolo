# 서버 이관 안내

노트북(RTX 3050 4GB)에서 하던 작업을 **Ubuntu 22.04 / i9-11900 / RTX 3090 24GB** 서버로 옮기기 위한 문서.
이 파일만 따라가면 같은 상태를 재현하고 다음 실험으로 넘어갈 수 있다.

작성 2026-09-09 · 정서인(비전)

---

## 0. 30초 요약

**지금 무엇을 하고 있나** — 항공 하향 영상에서 사람을 찾고, 나아가 **쓰러졌는지**를 판별하려 한다.

**어디까지 왔나**

| | |
|---|---|
| 운용 모델 | `yolov8s_pose3_sn_freeze.pt` (3클래스: person / fallen / ambiguous) |
| 자세 성능 | SARD test 쓰러짐 재현율 **0.974** |
| 탐지 강건성 | NOMAD 0.589 · WiSARD 0.804 · Okutama(미학습) 0.682 |
| 속도 | 1280×720 에서 20.6 ms (48.6 fps) |

**무엇이 막혀 있나** — 도메인이 바뀌면 자세 판별이 무너진다(Okutama 0.091).
NOMAD·Okutama 자세 라벨이 활동 구간에서 변환한 것이라 품질에 천장이 있는 것으로 추정된다.

**서버에서 할 첫 번째 일** — 5절의 실험 A(회전 증강). 무료이고 물리적으로 옳은 수정인데 아직 안 했다.

---

## 1. 운용 조건 (모든 판단의 기준)

```
카메라   IMX415 + 5.5mm 렌즈
화각     수평 54.0° / 대각 60.7°   ← 코드가 쓰는 건 수평
해상도   1920 × 1080
시점     고정 하향 90° (짐벌 없음)
순찰고도 16 m (시뮬레이터) · 운용 16~30 m
```

지상폭 `W = 2 × 고도 × tan(27°)` · `GSD = W / 1920`

| 고도 | 지상폭 | 선 자세 | 누운 자세 |
|---|---|---|---|
| 16 m | 16.3 m | 59 px | 200 px |
| 25 m | 25.5 m | 38 px | 128 px |
| 30 m | 30.6 m | 31 px | 107 px |

**주의** — `imgsz 960` 으로 학습하면 1920 입력이 레터박스로 **절반**이 된다.
30 m 선 자세 31 px 이 네트워크 입력에서 **15.5 px** 로, SRS 의 탐지 하한 20 px 아래다.
서버 24GB 에서는 `imgsz 1280` 이 가능하다 (20.7 px).

---

## 2. 서버 준비

```bash
git clone https://github.com/tim9985/drone_yolo.git
cd drone_yolo

conda create -n drone python=3.10 -y && conda activate drone
# CUDA 빌드 확인 필수 — CPU 빌드가 설치되기 쉽다
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python pyyaml
python -c "import torch; print(torch.__version__, torch.cuda.is_available(),
                              torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` 가 False 면 CPU 빌드다. 지우고 인덱스 지정해 다시 깔 것.

### 노트북 설정을 그대로 쓰지 말 것

기존 값은 **RTX 3050 4GB** 기준이라 3090 에서는 크게 낭비된다.

| 인자 | 노트북 | 서버(3090 24GB · i9-11900 · 64GB) |
|---|---|---|
| `--batch` | 6 | **16~32** (imgsz 960) · **8~16** (imgsz 1280) |
| `workers` | 4 | **8** (8코어 16스레드) |
| `cache` | False | `'disk'` 권장. `'ram'` 은 9,008장 디코딩에 ~24GB 필요 |

**단, 실험 A 는 batch 6 · lr0 0.001 을 그대로 쓴다.** 8차와 비교하는 게 목적이라
회전 인자 외의 변수를 바꾸면 원인을 못 가린다. 3090 이면 batch 6 으로도
에폭당 5.6분 → 2분 안쪽으로 줄어든다. batch 를 키울 거면 **별도 실행**으로 분리할 것.

batch 를 올릴 때는 학습률도 함께 봐야 한다. 우리가 `lr0 0.001` 을 쓰는 이유는
**사전학습 특징 보존**이라, 선형 스케일링(batch 6→32 이면 lr ×5.3)을 그대로 적용하면
파국적 망각이 다시 온다. batch 32 라면 `lr0 0.002` 정도까지만.

### JupyterLab 에서 돌릴 때

**노트북 셀에서 학습을 돌리지 말 것.** 커널이 재시작되거나 브라우저 연결이 끊기면
3시간짜리 학습이 사라진다. JupyterLab 터미널에서 `tmux` 를 쓴다.

```bash
tmux new -s train
conda activate drone
python train_person.py ...        # 여기서 실행
# Ctrl+B, D 로 분리 — 브라우저를 닫아도 계속 돈다
tmux attach -t train              # 다시 붙기
nvidia-smi -l 5                   # 다른 창에서 GPU 확인
```

---

## 3. 데이터 — 재배포 불가, 각자 받아야 한다

`data/` 는 `.gitignore` 에 있다. NOMAD·WiSARD·Okutama 는 연구용 라이선스라 재배포하지 않는다.

| 데이터셋 | 출처 | 라이선스 | 우리 용도 |
|---|---|---|---|
| **SARD** | Sambolek & Ivasic-Kos 2021 · IEEE DataPort `10.21227/ahxm-k331` (Roboflow 재배포본 사용) | CC BY 4.0 | **자세 학습의 핵심** — 사람이 직접 라벨링 |
| NOMAD | 원 배포처 | 연구용 | 배우 100명 · 여름 농장 |
| WiSARD | 원 배포처 | 연구용 | 겨울·가을 산지 |
| Okutama-Action | okutama-action.org | CC BY-NC-SA 3.0 | **평가 전용** — 우리 고도와 일치 |

받은 뒤 전처리:

```bash
python nomad_prep.py
python wisard_prep.py
python okutama_align.py --videos <영상id들> --range 60   # 프레임 오프셋 추정 (필수)
python okutama3_prep.py --videos 2.2.10 1.2.10 2.1.2 ...  # 평가셋 생성
python make_configs.py    # ★ configs/*.yaml 의 절대경로를 이 서버 기준으로 재생성
```

**`make_configs.py` 를 빼먹으면 안 된다.** 저장소의 yaml 에는 Windows 절대경로가 박혀 있다.

각 데이터셋의 **정확한 출처·다운로드 절차·기대 폴더 구조**는 [`DATASETS.md`](DATASETS.md) 에 있다.

---

## 4. 가중치 이관

`weights/` 도 `.gitignore` 에 있다. 아래 두 개만 있으면 재현 가능하다.

| 파일 | 용도 |
|---|---|
| `yolov8s_stage1_all.pt` | **모든 실험의 시작 가중치.** NOMAD+WiSARD 로 학습한 탐지 모델 |
| `yolov8s_pose3_sn_freeze.pt` | 현재 운용 모델 (3클래스) |

노트북에서 직접 복사하거나(scp), 없으면 `stage1_all` 부터 재학습해야 한다(수 시간).

**각 가중치가 어디서 왔고 무엇으로 학습됐는지는 [`WEIGHTS.md`](WEIGHTS.md) 에 있다.**

---

## 5. 다음 실험 — 우선순위대로

### A. 회전 증강 ⭐ 먼저 할 것

**현재 `degrees=10.0`, `flipud=0.0` 인데 하향 시점에서는 물리적으로 틀렸다.**
드론이 요잉하면 장면 전체가 돌고, 누운 사람은 어느 방향으로든 누울 수 있다.
화면에 "위쪽"이 없다. 그런데 ±10° 만 돌려 학습해서, 부족한 `fallen` 2,755개가
방향별로 쪼개졌다.

`train_person.py` 의 stage1 증강 프로필을 고친다.

```python
extra = dict(hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
             degrees=180.0,   # 10.0 에서
             flipud=0.5,      # 0.0 에서
             translate=0.15, scale=0.5, fliplr=0.5, mosaic=1.0)
```

```bash
python make_pose3_dataset.py --wisard exclude
python train_person.py --stage 1 --data configs/data_pose3_sn.yaml \
  --weights weights/yolov8s_stage1_all.pt --name pose3_rot \
  --epochs 35 --imgsz 960 --batch 6 --lr0 0.001 --patience 10 --freeze 10   # 8차와 동일 조건
python eval_pose3.py --weights weights/yolov8s_pose3_sn_freeze.pt \
                               runs_person/pose3_rot/weights/best.pt
```

**비교 기준** — 8차 `pose3_sn_freeze`: SARD test 쓰러짐 0.974 · NOMAD 탐지 0.589 · WiSARD 0.804 · Okutama 0.682

### B. imgsz 1280

24GB 면 가능하다. 1절의 픽셀 계산이 근거다. A 와 함께 걸어도 된다.

```bash
python train_person.py --stage 1 --data configs/data_pose3_sn.yaml \
  --weights weights/yolov8s_stage1_all.pt --name pose3_1280 \
  --epochs 35 --imgsz 1280 --batch 12 --lr0 0.001 --patience 10 --freeze 10
```

### C. 동결 범위 탐색

`--freeze 10` 은 백본만 얼린다. 넥·헤드(54.4%)는 여전히 SARD·NOMAD 만 보며
재조정돼 탐지가 −10% 남았다. **얼릴수록 안 잊지만 못 배운다** — 균형점을 아직 안 찾았다.

```bash
for f in 0 6 10 15 22; do
  python train_person.py --stage 1 --data configs/data_pose3_sn.yaml \
    --weights weights/yolov8s_stage1_all.pt --name pose3_f$f \
    --epochs 25 --imgsz 960 --batch 16 --lr0 0.001 --freeze $f   # 탐색이라 batch 키워도 됨
done
```

### D. 2단계 구조 (탐지 → 원본 해상도 크롭 분류)

자세에는 이게 구조적으로 맞다. 탐지는 960 에서 하고 **분류기는 원본 크롭**을 받는다.
16 m 에서 누운 사람은 1920 프레임에서 200 px 인데 분류기가 그 200 px 를 다 본다.
해상도 절충이 사라진다.

2·3차 시도가 이 구조였고 F1 0.47 에 그쳤지만, 그때는 **오염된 NOMAD 라벨만** 썼고
SARD 가 없었다. 지금은 SARD 깨끗한 라벨이 있다. `train_pose_cls.py` 를 SARD 로 다시.

### E. Copy-Paste (보류 중, 자산은 준비됨)

`extract_instances.py` → `dedup_instances.py` → `copy_paste.py` → `label_review.py`.
인스턴스 은행 1,116개를 만들어 뒀으나 **원천 인물 다양성이 낮아**(SARD ~8명,
Okutama ~6명, NOMAD ~30명) 보류했다. 배경은 곱할 수 있으나 사람은 만들지 못한다.
겨울·농장에 쓰러진 사람처럼 **우리에게 0개인 조합**이 필요할 때 재개할 것.

---

## 6. 하지 말 것 — 이미 확인된 막다른 길

| 시도 | 결과 | 근거 |
|---|---|---|
| **마스크 기하로 자세 판별** | **AUC 0.342~0.387** — 방향이 반대 | NOMAD 에서 서 있는 사람이 **더** 길쭉하다(중앙 2.39 대 1.96). 걷는 사람은 팔다리·그림자가 늘어지고 누운 사람은 웅크려 뭉툭해진다. OBB 도 같은 이유로 무의미 |
| 관측 해상도 키우기 | 6차: 97→200 px 로 2배 키워도 재현율 0.147 | 문제는 해상도가 아니라 **라벨**이었다 |
| SARD `not_defined` 배경 처리 | 9차: 오탐 −58% 로 진단은 맞았으나 NOMAD 발견율 −0.086 | 미탐지 비용이 오탐 비용보다 크다 |
| WiSARD 를 자세 학습에 포함 | 미실행 · 설계 단계에서 기각 | WiSARD `person` 은 **탐지** 라벨("사람이 있다")이고 우리 `person` 은 **자세** 라벨("서 있다"). 뜻이 다르다 |
| 낮은 학습률만으로 망각 방지 | 7차: lr 0.001 인데도 NOMAD 0.667→0.290 | 속도만 늦출 뿐 방향은 그대로. **동결이 필요하다** |

---

## 7. 겪은 함정

| 함정 | 증상 | 대응 |
|---|---|---|
| **Okutama 프레임 오프셋** | 상자가 사람에서 빗나감. 데이터 손상으로 오인함 | 영상마다 0~21 프레임 어긋나 있다. `okutama_align.py` 필수 |
| **H.264 임의 위치 탐색** | `cap.set(POS_FRAMES)` 로 건너뛰면 프레임이 어긋남 | 순차 디코딩만 쓸 것 |
| **상대경로 yaml** | ultralytics 가 **yaml 이 있는 configs/ 기준**으로 풀어서 죽음 | 항상 절대경로. `Path.resolve()` |
| **파일 탐색 패턴** | `Drone*/*/*.mp4` 가 한 단계 깊은 30편을 놓침 (10편인 줄 앎) | `rglob` |
| 조기종료 기준 | mAP50 이 아니라 `fitness = 0.1·mAP50 + 0.9·mAP50-95` | 정체 판단할 때 fitness 로 볼 것 |
| 검증셋 표본 | 도메인당 70장 표본이 WiSARD 0.897 을 0.883 으로 과대평가 | 최종 판정은 전량으로 |
| `wmic` | Windows 11 에서 제거됨. 대기 루프가 즉시 통과해 학습 중 평가가 시작됨 | 프로세스 목록 대신 **산출물 파일** 존재로 판단 |

---

## 8. 평가 — 항상 같은 잣대로

```bash
python eval_pose3.py --weights <모델들...>
```

합격 기준 4개(NOMAD 탐지 ≥0.60 · WiSARD ≥0.85 · SARD 쓰러짐 ≥0.90 · NOMAD 쓰러짐)를
한 번에 재고, 모델의 클래스 수(1/3/6)를 보고 매핑을 자동으로 고른다.
**비교 대상을 같은 코드로 다시 재지 않으면 공정한 비교가 아니다.**

> NOMAD 쓰러짐 목표 0.50 은 근거 없이 정한 값이었다. 그 도메인 특화 학습 모델의
> 최고가 0.177 이다. 현재 0.227 은 이미 최고 기록이다.

실영상 확인:

```bash
python video_infer.py --video <mp4> --weights <pt> --stride 3   # 일괄, 파일 출력
python live_view.py --source <mp4> --weights <pt>               # 대화형 창
```

**정답셋 지표로는 안 잡히는 실패가 있다.** `ambiguous` 가 벤치·가방에 붙는 문제는
mAP 로는 안 보였고 프레임을 눈으로 열어서야 드러났다.

---

## 9. 문서

전체 맥락은 <https://github.com/tim9985/Capstone> `보고서/비전_문서/` 에 있다.

| 문서 | 내용 |
|---|---|
| 01 | 비전 요구사항 명세 (SRS) — 05절에 자세 판정 현황 |
| 05 | 자세판별 선행조사 |
| 06 | 자세 라벨 설계 논의 — 3클래스가 왜 이렇게 정의됐는지 |
| **07** | **자세 3클래스 실험 기록 — 8·9차 전체 수치** |
