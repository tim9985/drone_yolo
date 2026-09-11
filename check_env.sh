#!/usr/bin/env bash
# check_env.sh — 서버 재부팅 직후 상태를 한 번에 점검한다
#
# 왜 필요한가
#   재부팅하면 tmux 세션이 전부 죽는다. 돌고 있던 다운로드가 어디서 멈췄는지,
#   드라이버가 제대로 올라왔는지, 무엇을 다시 해야 하는지를 모른 채 이어가면
#   반쯤 받은 zip 을 풀거나, CPU 로 학습을 돌리는 사고가 난다.
#
# 실행: bash check_env.sh
set -u
cd "$(dirname "$0")"

ok()   { printf "  \033[32m[OK]\033[0m   %s\n" "$*"; }
warn() { printf "  \033[33m[확인]\033[0m %s\n" "$*"; }
bad()  { printf "  \033[31m[없음]\033[0m %s\n" "$*"; }
hdr()  { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

# 노트북에서 8차 기준선을 만든 버전. 다르면 수치 비교가 흔들린다.
WANT_ULTRA="8.4.102"

hdr "1. GPU·드라이버"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  CUDA_MAX=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
  ok "드라이버 $DRV · $GPU"
  ok "드라이버가 지원하는 CUDA 최대 $CUDA_MAX"
else
  bad "nvidia-smi 실패 — 드라이버가 안 올라왔다. 이 상태로는 학습 불가"
fi

hdr "2. conda 환경"
if command -v conda >/dev/null 2>&1; then
  if conda env list | grep -qE '^drone\s'; then ok "drone 환경 있음"
  else bad "drone 환경 없음 — conda create -n drone python=3.10 -y"; fi
else
  bad "conda 명령 없음 — 셸 재시작 또는 ~/miniconda3/bin/conda init"
fi

hdr "3. 파이썬 패키지 (drone 환경)"
PYBIN=""
for c in "$HOME/miniconda3/envs/drone/bin/python" "$HOME/anaconda3/envs/drone/bin/python"; do
  [ -x "$c" ] && PYBIN="$c" && break
done
if [ -n "$PYBIN" ]; then
  "$PYBIN" - "$WANT_ULTRA" <<'PY'
import sys
want = sys.argv[1]
def line(tag, msg): print(f"  {tag} {msg}")
try:
    import torch
    cuda = torch.cuda.is_available()
    tag = "\033[32m[OK]\033[0m  " if cuda else "\033[31m[없음]\033[0m"
    line(tag, f"torch {torch.__version__} · CUDA 사용 {cuda}"
         + (f" · {torch.cuda.get_device_name(0)}" if cuda else " ← CPU 빌드거나 드라이버 불일치"))
except Exception as e:
    line("\033[31m[없음]\033[0m", f"torch 불러오기 실패: {e}")
try:
    import ultralytics
    v = ultralytics.__version__
    tag = "\033[32m[OK]\033[0m  " if v == want else "\033[33m[확인]\033[0m"
    line(tag, f"ultralytics {v}" + ("" if v == want else f" ← 기준선은 {want}. 수치 비교가 흔들린다"))
except Exception:
    line("\033[31m[없음]\033[0m", "ultralytics 미설치")
for m in ("cv2", "numpy", "yaml", "olefile"):
    try:
        mod = __import__(m)
        line("\033[32m[OK]\033[0m  ", f"{m} {getattr(mod, '__version__', '')}")
    except Exception:
        line("\033[33m[확인]\033[0m", f"{m} 미설치")
PY
else
  bad "drone 환경의 python 을 못 찾음"
fi

hdr "4. 저장소"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git fetch -q origin 2>/dev/null
  BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "?")
  if [ "$BEHIND" = "0" ]; then ok "최신 ($(git log --oneline -1))"
  else warn "원격보다 ${BEHIND}커밋 뒤 — git pull"; fi
else
  bad "git 저장소가 아니다 — 폴더만 만든 상태일 수 있다"
fi

hdr "5. 가중치"
for w in yolov8s_stage1_all.pt yolov8s_pose3_sn_freeze.pt; do
  if [ -f "weights/$w" ]; then ok "$w ($(du -h "weights/$w" | cut -f1))"
  else bad "$w — 노트북에서 scp. stage1_all 이 모든 실험의 시작점"; fi
done

hdr "6. 원본 데이터 (재부팅으로 다운로드가 끊겼을 수 있다)"
R=data/raw
chk_dir() {  # 이름 경로 기대하는_하위
  local name=$1 path=$2 sub=${3:-}
  if [ -d "$path" ]; then
    local sz n
    sz=$(du -sh "$path" 2>/dev/null | cut -f1)
    n=$(find "$path" -type f 2>/dev/null | wc -l)
    if [ -n "$sub" ] && [ ! -e "$path/$sub" ]; then
      warn "$name  $sz · 파일 $n개 · '$sub' 없음 → 받다 끊겼을 가능성"
    else
      ok "$name  $sz · 파일 $n개"
    fi
  else
    bad "$name  ($path)"
  fi
}
chk_dir "NOMAD 라벨"    "$R/NOMAD/labels"
chk_dir "NOMAD 이미지"  "$R/NOMAD/images"
[ -f "$R/NOMAD/activityLabels.json" ] && ok "NOMAD activityLabels.json" \
  || bad "NOMAD activityLabels.json — 자세 라벨의 근거. rclone copy gdrive:annotations"
if [ -d "$R/NOMAD/images" ]; then
  NA=$(ls "$R/NOMAD/images" 2>/dev/null | wc -l)
  [ "$NA" -ge 100 ] && ok "NOMAD 배우 폴더 $NA개" || warn "NOMAD 배우 폴더 $NA/100 — 이어받기 필요"
fi
chk_dir "WiSARD"        "$R/WiSARD"
ls "$R"/WiSARDv1.zip* >/dev/null 2>&1 && warn "WiSARDv1.zip 이 남아 있다 — 받다 끊긴 것일 수 있다. 크기가 40.54GB 인지 확인"
chk_dir "SARD"          "$R/sard2/search-and-rescue-2" "test"
chk_dir "Okutama"       "$R/okutama" "Labels"

hdr "7. 산출 데이터셋"
for d in pose3_sn okutama3 wisard; do
  if [ -d "data/det/$d" ]; then ok "data/det/$d"
  else warn "data/det/$d — 전처리 전"; fi
done
if [ -d data/det/pose3_sn/labels/train ]; then
  "${PYBIN:-python3}" - <<'PY'
from pathlib import Path
from collections import Counter
c = Counter(); n = 0
for f in Path("data/det/pose3_sn/labels/train").glob("*.txt"):
    n += 1
    for ln in f.read_text().splitlines():
        p = ln.split()
        if p: c[int(p[0])] += 1
got = (n, c[0], c[1], c[2]); want = (9008, 7562, 2755, 1731)
tag = "\033[32m[OK]\033[0m  " if got == want else "\033[33m[확인]\033[0m"
print(f"  {tag} pose3_sn train {n}장 · person {c[0]} · fallen {c[1]} · ambiguous {c[2]}")
if got != want:
    print(f"         기대값 {want[0]} / {want[1]} / {want[2]} / {want[3]} — 다르면 8차와 비교 불가")
PY
fi

hdr "8. 디스크·세션"
df -h . | tail -1 | awk '{printf "  남은 공간 %s / 전체 %s (%s 사용)\n", $4, $2, $5}'
if command -v tmux >/dev/null 2>&1; then
  S=$(tmux ls 2>/dev/null | wc -l)
  [ "$S" -eq 0 ] && warn "tmux 세션 0개 — 재부팅으로 다운로드·학습이 모두 종료됐다" \
                 || ok "tmux 세션 ${S}개"
fi
echo
