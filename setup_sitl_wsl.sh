#!/bin/bash
# setup_sitl_wsl.sh — WSL2 Ubuntu에서 ArduPilot SITL 자동 설치 (M3)
# 사용법: WSL Ubuntu 터미널에서
#   bash setup_sitl_wsl.sh
# 소요: 15~30분 (네트워크·PC 성능에 따라)
set -e

echo "=== [1/5] 시스템 패키지 설치 ==="
sudo apt update
sudo apt install -y git python3 python3-pip python3-dev python3-setuptools \
    build-essential cmake g++ libxml2-dev libxslt1-dev \
    python3-matplotlib python3-tk python3-lxml

echo "=== [2/5] ArduPilot 소스 클론 ==="
cd ~
if [ ! -d "ardupilot" ]; then
    git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
else
    echo "ardupilot 폴더가 이미 존재 — 클론 생략"
fi
cd ~/ardupilot

echo "=== [3/5] 의존성 설치 스크립트 실행 ==="
Tools/environment_install/install-prereqs-ubuntu.sh -y
source ~/.profile

echo "=== [4/5] SITL 빌드 (반드시 저장소 루트에서) ==="
./waf configure --board sitl
./waf copter

echo "=== [5/5] 검증 안내 ==="
echo ""
echo "빌드 완료! 아래 명령으로 SITL 단독 테스트 (M2):"
echo "  cd ~/ardupilot/ArduCopter"
echo "  ../Tools/autotest/sim_vehicle.py --console --map"
echo ""
echo "MAVProxy 콘솔에서: mode GUIDED → arm throttle → takeoff 10"
echo "지도에서 드론이 상승하면 M2 통과."
echo ""
echo "AirSim 연동 실행 (M3.5, AirSim Play를 먼저 누른 뒤):"
echo "  cd ~/ardupilot/ArduCopter"
echo "  ../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter --console --map"
