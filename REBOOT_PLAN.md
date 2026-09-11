# 재부팅 후 진행 계획

서버 재부팅 직후부터 첫 실험 결과까지의 순서. **각 단계는 앞 단계가 통과해야 넘어간다.**

작성 2026-09-11

> **2026-09-12 상태** — 서버는 이 계획 대신 `EXPERIMENTS.md` 의 E1~E8(COCO 사전학습에서
> 새로 시작하는 1클래스 탐지 스윕)을 진행 중이다. 회전 증강(7단계)은 그쪽에 이미 반영됐다.
> 이 문서는 **3클래스 자세 트랙(pose3)을 서버에서 재개할 때** 쓴다. 전체 맥락은 `CLAUDE.md`.

---

## 먼저 알아둘 것

**재부팅하면 tmux 세션이 전부 죽는다.** 받고 있던 다운로드는 중간에 끊긴 상태다.

| 도구 | 끊겼을 때 |
|---|---|
| `rclone copy` | **같은 명령을 다시 돌리면 받은 파일은 건너뛴다.** 안전하다 |
| `gdown` (WiSARD 40GB zip) | **반쯤 받은 zip 이 남는다.** 풀면 깨진다. 크기 확인 필수 |

---

## 0단계 — 상태 점검 (5분)

```bash
cd ~/drone_dev && git pull
bash check_env.sh
```

`[없음]` 이 뜬 항목부터 아래 단계에서 해결한다. 전부 `[OK]` 면 5단계로 건너뛴다.

---

## 1단계 — GPU 드라이버

**재부팅 이유가 드라이버 설치였다면 여기서 성패가 갈린다.**

```bash
nvidia-smi
```

| 결과 | 조치 |
|---|---|
| 표가 나오고 `RTX 3090` 이 보인다 | 통과. `CUDA Version` 값을 기억해 둔다 (2단계에서 쓴다) |
| `command not found` | 드라이버 미설치 → `sudo ubuntu-drivers install` 후 재부팅 |
| `couldn't communicate with the NVIDIA driver` | 커널 모듈 불일치 → `sudo apt install --reinstall nvidia-driver-XXX` 후 재부팅 |

> 학과 서버라 `sudo` 가 없으면 관리자에게 요청해야 한다. 드라이버는 사용자 권한으로 못 깐다.

---

## 2단계 — PyTorch 를 드라이버에 맞춰 설치

**드라이버가 지원하는 CUDA 버전보다 높은 빌드를 깔면 GPU 를 못 잡는다.**
`nvidia-smi` 우측 상단의 `CUDA Version` 이 **상한**이다.

| `nvidia-smi` 의 CUDA Version | 설치할 빌드 |
|---|---|
| 13.0 이상 | `cu130` (노트북과 동일) |
| 12.4 ~ 12.x | `cu124` |
| 12.1 ~ 12.3 | `cu121` |

```bash
conda activate drone      # 없으면: conda create -n drone python=3.10 -y && conda activate drone

# 위 표에서 고른 빌드로
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ★ ultralytics 는 반드시 노트북과 같은 버전으로 고정
pip install ultralytics==8.4.102 opencv-python pyyaml olefile gdown

python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
```

**기대 출력** — `2.x.x+cu124 True 8.4.102`

### 왜 ultralytics 버전을 고정하나

8차 기준선(SARD 쓰러짐 0.974 · NOMAD 0.589 · WiSARD 0.804 · Okutama 0.682)은
**8.4.102** 로 만들었다. 버전이 바뀌면 증강 기본값·손실 계산·평가 방식이 달라져,
실험 A 에서 수치가 변해도 **회전 증강 때문인지 버전 때문인지 가릴 수 없다.**

노트북 환경 (기준)

```
python       3.10.20
torch        2.13.0+cu130
ultralytics  8.4.102
opencv       5.0.0
numpy        2.2.6
```

torch 는 드라이버에 맞춰 달라져도 괜찮다. **ultralytics 만은 맞출 것.**

---

## 3단계 — 끊긴 다운로드 재개

### NOMAD — 그대로 다시 돌린다

```bash
tmux new -s dl
cd ~/drone_dev/data/raw
export RCLONE_DRIVE_ROOT_FOLDER_ID=1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg

rclone copy gdrive:annotations ./NOMAD -P
rclone copy gdrive:labels ./NOMAD/labels -P \
  --transfers 32 --checkers 32 --drive-pacer-min-sleep 10ms --drive-pacer-burst 200
rclone copy gdrive:images ./NOMAD/images -P \
  --transfers 16 --checkers 32 --drive-pacer-min-sleep 10ms --drive-pacer-burst 200
```

`export` 는 재부팅으로 사라졌다. **다시 넣어야 한다.** 매번 넣기 싫으면:

```bash
echo 'export RCLONE_DRIVE_ROOT_FOLDER_ID=1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg' >> ~/.bashrc
```

완료 확인 — `ls NOMAD/images | wc -l` 이 **100**

### WiSARD — 반쯤 받은 zip 부터 확인

```bash
ls -l ~/drone_dev/data/raw/WiSARDv1.zip 2>/dev/null
```

| 상태 | 조치 |
|---|---|
| 파일 없음 | 새로 받는다 |
| 40.54 GB 미만 | **이어받는다** — `gdown --continue` |
| 40.54 GB | `unzip -t` 로 무결성 확인 후 푼다 |

```bash
cd ~/drone_dev/data/raw
gdown --continue 1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht -O WiSARDv1.zip
unzip -tq WiSARDv1.zip && unzip -q WiSARDv1.zip -d WiSARD && rm WiSARDv1.zip
```

`unzip -t` 에서 오류가 나면 **절대 풀지 말고** 지운 뒤 다시 받는다.

### SARD

```bash
pip install roboflow
python - <<'PY'
from roboflow import Roboflow
Roboflow(api_key="<키>").workspace("rescuedby").project("sard-peykp-lxuf9") \
  .version(1).download("yolov8", location="data/raw/sard2")
PY
```

### Okutama

공식 Dropbox 가 막혀 있다. 노트북에서 복사한다 (9.6 GB). **평가 전용이라 학습에는 없어도 된다.**

---

## 4단계 — 가중치

노트북 MobaXterm SFTP 패널로 `~/drone_dev/weights/` 에 넣는다.

```
yolov8s_stage1_all.pt        64 MB  ← 모든 실험의 시작점 (옵티마이저 상태 포함)
yolov8s_pose3_sn_freeze.pt   21.5 MB  ← 8차 기준선
```

---

## 5단계 — 전처리

```bash
cd ~/drone_dev && conda activate drone

python nomad_prep.py
python wisard_prep.py
python make_pose3_dataset.py --wisard exclude
python make_configs.py        # ★ yaml 절대경로를 이 서버 기준으로
```

### 검증 — 여기서 틀리면 멈출 것

```bash
bash check_env.sh      # 7번 항목을 본다
```

**기대값 — `pose3_sn train 9008장 · person 7562 · fallen 2755 · ambiguous 1731`**

전처리 스크립트는 시드를 고정하고 셔플 전에 정렬하므로, **원본이 같으면 노트북과
똑같은 데이터셋이 나온다.** 숫자가 다르면 원본 다운로드가 불완전한 것이다.

---

## 6단계 — 기준선 재현 ★ 실험 전에 반드시

**새 실험을 돌리기 전에, 기존 8차 가중치를 서버에서 다시 잰다.**

```bash
python eval_pose3.py --weights weights/yolov8s_pose3_sn_freeze.pt
```

| 지표 | 노트북 기록 | 허용 오차 |
|---|---:|---:|
| SARD test 쓰러짐 재현율 | 0.974 | ±0.01 |
| NOMAD 탐지 발견율 | 0.589 | ±0.01 |
| WiSARD 탐지 발견율 | 0.804 | ±0.01 |

> Okutama(0.682)는 원본이 서버에 없으면 측정이 빠진다. 그래도 된다.

**이 단계를 건너뛰면 안 된다.** 서버에서 같은 가중치가 다른 수치를 내면 환경이
다르다는 뜻이고, 그 상태에서 실험 A 가 "개선됐다"고 나와도 믿을 수 없다.

- **일치** → 7단계로
- **어긋남** → 원인부터 찾는다. 대개 ultralytics 버전 또는 데이터셋 불일치

---

## 7단계 — 실험 A: 회전 증강

`train_person.py` 의 stage1 증강 프로필에서 두 줄을 고친다.

```python
degrees=180.0,   # 10.0 에서 — 하향 시점엔 위쪽이 없다
flipud=0.5,      # 0.0 에서
```

```bash
tmux new -s train
python train_person.py --stage 1 --data configs/data_pose3_sn.yaml \
  --weights weights/yolov8s_stage1_all.pt --name pose3_rot \
  --epochs 35 --imgsz 960 --batch 6 --lr0 0.001 --patience 10 --freeze 10
# Ctrl+B, D 로 분리
```

`batch 6` 은 일부러 그대로 둔다. 8차와 **회전 인자만** 다르게 해야 원인을 가린다.

끝나면:

```bash
python eval_pose3.py --weights weights/yolov8s_pose3_sn_freeze.pt \
                               runs_person/pose3_rot/weights/best.pt
```

---

## 예상 소요

| 단계 | 시간 |
|---|---|
| 0~2 점검·드라이버·패키지 | 30분 |
| 3 다운로드 | **수 시간** (NOMAD images · WiSARD 40GB) |
| 4 가중치 | 5분 |
| 5 전처리 | 30분~1시간 |
| 6 기준선 재현 | 20분 |
| 7 실험 A | 1.5~2시간 (3090 기준) |

3단계가 병목이다. **NOMAD 와 WiSARD 를 tmux 창 두 개에서 동시에** 받으면 줄어든다.
