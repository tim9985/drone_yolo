"""
run_experiment.py — 드론 캡스톤 실험 전자동 오케스트레이터

사람 개입 없이 아래 전 과정을 수행한다:
  [1] 사전 점검      → conda/GPU/필수파일 확인, 잔여 프로세스 정리
  [2] 언리얼 실행    → 바로가기에서 경로 추출 후 실행 (자동/에디터 모드)
  [3] SITL 실행      → WSL에서 ArduPilot SITL 원격 구동
  [4] MAVLink 연결   → pymavlink로 연결 + Arm + Takeoff (MAVProxy 타이핑 대체)
  [5] 실험 실행      → --benchmark / --patrol / --autonomous / --collect
  [6] 안전 종료      → RTL/LAND → disarm → 언리얼 종료 → SITL 종료 (순서 엄수)

사용법:
  python run_experiment.py --benchmark
  python run_experiment.py --patrol
  python run_experiment.py --autonomous
  python run_experiment.py --collect 200
  python run_experiment.py --editor --patrol   (에디터 수동 모드, 데모 녹화용)
  python run_experiment.py --benchmark --no-cleanup  (디버깅용, 프로세스 정리 생략)
"""
import argparse
import csv
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    # PowerShell/cmd 콘솔 코드페이지(cp949 등)에 상관없이 한글 로그가 깨지지 않도록 강제
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
LNK_PATH = BASE_DIR / "Blocks_AirSim_Play.lnk"
WEIGHTS_DIR = BASE_DIR / "weights"
LOG_DIR = BASE_DIR / "logs"
COMPARISON_DIR = BASE_DIR / "comparison"
BENCHMARK_CSV = BASE_DIR / "metrics" / "benchmark_results.csv"

SITL_PARM = "/home/timothy/sitl_params.parm"
# --no-mavproxy 시 ArduPilot SITL은 GCS 링크로 기본 TCP:5760(로컬)을 연다(UDP:14550이 아님 — 그건
# MAVProxy의 --out 옵션으로만 열림). 따라서 1차는 TCP 5760, 실패 시 MAVProxy+--out으로 UDP 14550 폴백.
MAVLINK_TCP_NO_MAVPROXY = "tcp:127.0.0.1:5760"
MAVLINK_UDP_MAVPROXY = "udp:0.0.0.0:14550"
CRUISE_ALT = 8.0
FPS_TARGET = 5.0


# ═══════════════════════════════════════════════════════════════════════
# 로깅
# ═══════════════════════════════════════════════════════════════════════
def setup_logger():
    LOG_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{ts}.log"

    logger = logging.getLogger("run_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger, log_path


class StepTimer:
    """단계별 시작/종료 시각 및 소요시간을 로그에 남기는 컨텍스트 매니저."""

    def __init__(self, logger, name):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"▶ [{self.name}] 시작")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.t0
        if exc_type is None:
            self.logger.info(f"✔ [{self.name}] 완료 ({elapsed:.1f}s)")
        else:
            self.logger.error(f"✘ [{self.name}] 실패 ({elapsed:.1f}s): {exc}")
        return False  # 예외 전파


# ═══════════════════════════════════════════════════════════════════════
# [1] 사전 점검
# ═══════════════════════════════════════════════════════════════════════
def check_conda_env(logger):
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix = sys.prefix
    is_drone = env == "drone" or "envs" + os.sep + "drone" in prefix
    if not is_drone:
        logger.warning(f"conda 환경이 'drone'이 아닌 것으로 보임 (현재: {env or prefix}). "
                        f"'conda activate drone' 후 재실행을 권장.")
    else:
        logger.info(f"conda 환경 확인 완료: drone ({prefix})")
    return is_drone


def check_gpu(logger):
    try:
        import torch
        ok = torch.cuda.is_available()
        if ok:
            logger.info(f"GPU 확인: {torch.cuda.get_device_name(0)} (CUDA 사용 가능)")
        else:
            logger.warning("CUDA 사용 불가 — CPU로 추론 (FPS 저하 예상)")
        return ok
    except Exception as e:
        logger.warning(f"GPU 확인 실패: {e}")
        return False


def check_files(logger):
    required = [
        LNK_PATH,
        WEIGHTS_DIR / "yolov8s_visdrone.pt",
        WEIGHTS_DIR / "yolov8s_stage1_all.pt",   # 정찰·자율탐색이 쓰는 학습 가중치
        BASE_DIR / "yolov8n.pt",
        BASE_DIR / "patrol_detect.py",
        BASE_DIR / "map_manager.py",
    ]
    missing = []
    for p in required:
        exists = p.exists()
        logger.info(f"파일 확인: {p.name} {'OK' if exists else 'MISSING'}")
        if not exists:
            missing.append(str(p))
    return len(missing) == 0, missing


def check_wsl_sitl(logger):
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-lc",
             "test -f ~/ardupilot/Tools/autotest/sim_vehicle.py && echo SITL_OK || echo SITL_MISSING"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
        ok = "SITL_OK" in result.stdout
        logger.info(f"WSL ArduPilot SITL 빌드 확인: {'OK' if ok else 'MISSING'}")
        return ok
    except Exception as e:
        logger.warning(f"WSL 확인 실패: {e}")
        return False


def cleanup_processes(logger):
    subprocess.run(["taskkill", "/F", "/IM", "UnrealEditor.exe"],
                    capture_output=True)
    subprocess.run(
        ["wsl", "-e", "bash", "-lc", "pkill -f arducopter 2>/dev/null; pkill -f sim_vehicle.py 2>/dev/null; pkill -f mavproxy 2>/dev/null; true"],
        capture_output=True,
    )
    logger.info("이전 실행의 잔여 프로세스(UnrealEditor, arducopter/sim_vehicle) 정리 완료")


def preflight(logger):
    check_conda_env(logger)
    check_gpu(logger)
    check_wsl_sitl(logger)
    ok, missing = check_files(logger)
    if not ok:
        raise RuntimeError(f"필수 파일 누락: {missing}")
    cleanup_processes(logger)


# ═══════════════════════════════════════════════════════════════════════
# [2] 언리얼 실행
# ═══════════════════════════════════════════════════════════════════════
def extract_shortcut(logger, lnk_path: Path):
    """PowerShell WScript.Shell로 .lnk의 TargetPath/Arguments/WorkingDirectory 추출.
    콘솔 코드페이지 문제를 피하려고 UTF-8 JSON 파일로 결과를 주고받는다."""
    tmp_out = BASE_DIR / "_lnk_info.json"
    ps_script = f'''
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut("{lnk_path}")
$obj = [PSCustomObject]@{{
    TargetPath = $lnk.TargetPath
    Arguments = $lnk.Arguments
    WorkingDirectory = $lnk.WorkingDirectory
}}
$obj | ConvertTo-Json | Out-File -FilePath "{tmp_out}" -Encoding utf8
'''
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, capture_output=True)
    data = json.loads(tmp_out.read_text(encoding="utf-8-sig"))
    tmp_out.unlink(missing_ok=True)
    logger.info(f"바로가기 추출: target={data['TargetPath']} args={data['Arguments']}")
    return data["TargetPath"], data["Arguments"], data["WorkingDirectory"]


def launch_unreal(logger, editor_mode: bool, settings_override: Path = None):
    target, args, workdir = extract_shortcut(logger, LNK_PATH)

    if settings_override is not None:
        # 바로가기의 -settings="C:\AirSim\settings.json" 을 지정 파일로 교체
        # (--synth: ComputerVision 모드 등 실험별 별도 설정 사용)
        import re as _re
        # 치환 문자열을 람다로 넘겨 백슬래시(\U 등)가 이스케이프로 해석되는 것 방지
        args = _re.sub(r'-settings="[^"]*"', lambda _m: f'-settings="{settings_override}"', args)
        logger.info(f"settings 교체: {settings_override}")

    if editor_mode:
        full_args = args
    else:
        full_args = f"{args} -game -windowed -ResX=1280 -ResY=720"

    cmdline = f'"{target}" {full_args}'
    logger.info(f"언리얼 실행 ({'에디터' if editor_mode else '자동(-game)'} 모드): {cmdline}")
    proc = subprocess.Popen(cmdline, cwd=workdir or None)

    if editor_mode:
        logger.info("에디터 모드입니다. 언리얼 에디터가 열리면 Play를 누르세요.")
        input(">>> Play를 누른 뒤 Enter를 눌러 계속 진행하세요...")
    return proc


def wait_airsim_ready(logger, timeout=1800):
    """AirSim RPC 서버가 응답할 때까지 폴링. 첫 실행은 셰이더 컴파일로 수십 분 걸릴 수 있음."""
    import cosysairsim as airsim

    t0 = time.time()
    last_log = 0.0
    attempt = 0
    while time.time() - t0 < timeout:
        attempt += 1
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            logger.info(f"AirSim 연결 확인 완료 ({time.time() - t0:.0f}s 소요, {attempt}회 시도)")
            return client
        except Exception:
            elapsed = time.time() - t0
            if elapsed - last_log > 30:
                logger.info(f"AirSim 대기 중... ({elapsed:.0f}s 경과 — 첫 실행 시 셰이더 컴파일로 수십 분 소요 가능)")
                last_log = elapsed
            time.sleep(5)
    raise TimeoutError(f"AirSim 연결 실패 (타임아웃 {timeout}s)")


# ═══════════════════════════════════════════════════════════════════════
# [3] SITL 실행 (WSL)
# ═══════════════════════════════════════════════════════════════════════
def launch_sitl(logger, no_mavproxy=True):
    if no_mavproxy:
        cmd = (
            "cd ~/ardupilot/ArduCopter && "
            "../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter "
            f"--add-param-file={SITL_PARM} --no-mavproxy"
        )
    else:
        # MAVProxy는 대화형 셸을 가정하므로 stdin이 없는 headless 실행에서는
        # EOF를 받자마자 즉시 종료해버린다(--daemon으로 방지).
        cmd = (
            "cd ~/ardupilot/ArduCopter && "
            "../Tools/autotest/sim_vehicle.py -v ArduCopter -f airsim-copter "
            f"--add-param-file={SITL_PARM} --out udp:127.0.0.1:14550 "
            '--mavproxy-args="--daemon"'
        )

    log_path = LOG_DIR / "sitl_output.log"
    logf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(["wsl", "-e", "bash", "-lc", cmd], stdout=logf, stderr=subprocess.STDOUT)
    logger.info(f"SITL 실행 시작 ({'--no-mavproxy' if no_mavproxy else 'MAVProxy+--out'} 방식, "
                f"PID={proc.pid}), 로그: {log_path}")
    return proc, logf


def stop_sitl(logger, sitl_proc, sitl_logf):
    try:
        subprocess.run(
            ["wsl", "-e", "bash", "-lc", "pkill -f arducopter 2>/dev/null; pkill -f sim_vehicle.py 2>/dev/null; pkill -f mavproxy 2>/dev/null; true"],
            capture_output=True,
        )
        if sitl_proc is not None:
            sitl_proc.terminate()
    except Exception as e:
        logger.warning(f"SITL 종료 중 오류(무시): {e}")
    finally:
        if sitl_logf:
            sitl_logf.close()


# ═══════════════════════════════════════════════════════════════════════
# [4] MAVLink 연결 및 이륙
# ═══════════════════════════════════════════════════════════════════════
def wait_vehicle_heartbeat(logger, drone, timeout=60):
    """SITL 부팅 초기에는 autopilot=INVALID인 '가짜' heartbeat가 먼저 도착해
    pymavlink가 target_system을 0으로 고정해버리는 경우가 있다(이후 모든 명령이
    씹혀서 motors_armed_wait()가 무한 대기하는 원인). target_system이 실제로
    잠길 때까지 heartbeat를 반복 수신한다."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        remaining = max(1, timeout - (time.time() - t0))
        msg = drone.wait_heartbeat(timeout=remaining)
        if drone.target_system != 0:
            return
        if msg is None:
            logger.info(f"[MAVLink] heartbeat 미수신 ({remaining:.0f}s 대기) — 링크에 트래픽 없음")
        else:
            logger.info("[MAVLink] heartbeat 수신했으나 target_system=0 (부팅 초기 autopilot=INVALID heartbeat) — 재대기")
    raise TimeoutError(f"유효한 비행체 heartbeat 수신 실패 (target_system이 계속 0, {timeout}s 초과)")


def wait_armed(logger, drone, timeout=20):
    """motors_armed_wait()는 타임아웃이 없어 Arm 실패 시 영원히 멈춘다.
    직접 폴링하여 timeout 내에 armed 플래그를 확인하지 못하면 False를 반환.
    COMMAND_ACK/STATUSTEXT를 함께 로깅해 거부 사유(원인)를 남긴다."""
    from pymavlink import mavutil
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = drone.recv_match(blocking=True, timeout=2)
        if msg is None:
            continue
        mtype = msg.get_type()
        if mtype == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            try:
                result_name = mavutil.mavlink.enums["MAV_RESULT"][msg.result].name
            except Exception:
                result_name = str(msg.result)
            logger.info(f"[MAVLink] ARM COMMAND_ACK: {result_name}")
        elif mtype == "STATUSTEXT":
            logger.info(f"[MAVLink] STATUSTEXT: {msg.text}")
        elif mtype == "HEARTBEAT" and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def connect_and_takeoff(logger, altitude, udp, retries=3):
    from pymavlink import mavutil

    last_err = None
    drone = None
    for attempt in range(1, retries + 1):
        try:
            if drone is not None:
                # 이전 시도의 소켓을 남겨두면(특히 TCP) SITL 쪽이 새 연결의
                # heartbeat 인식을 방해할 수 있어 재시도 전 반드시 정리한다.
                try:
                    drone.close()
                except Exception:
                    pass
                drone = None
            logger.info(f"[MAVLink] 연결 시도 {attempt}/{retries}: {udp}")
            drone = mavutil.mavlink_connection(udp)
            wait_vehicle_heartbeat(logger, drone, timeout=60)
            logger.info(f"[MAVLink] heartbeat 수신 (system={drone.target_system})")

            # raw TCP/UDP 직결에서는 ArduPilot이 위치 메시지를 스트리밍하지 않으므로
            # (MAVProxy가 평소 대신 해주던) 데이터 스트림 요청을 직접 보낸다.
            drone.mav.request_data_stream_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
            )

            # 파라미터 파일 미적용 대비 ARMING_CHECK 강제 0
            drone.mav.param_set_send(
                drone.target_system, drone.target_component,
                b"ARMING_CHECK", 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32,
            )
            time.sleep(1)

            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0, 1, 4, 0, 0, 0, 0, 0,  # GUIDED = 4
            )
            time.sleep(1)

            # SITL 부팅 직후에는 EKF/GPS 준비 전이라 arm이 거부된다(EKF origin·위치추정
            # 확보까지 통상 30~60s). 준비될 때까지 5초 간격으로 arm을 재전송한다.
            armed = False
            arm_deadline = time.time() + 90
            while time.time() < arm_deadline:
                drone.mav.command_long_send(
                    drone.target_system, drone.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 1, 21196, 0, 0, 0, 0, 0,  # param2=21196: prearm 체크 강제 우회
                )
                if wait_armed(logger, drone, timeout=5):
                    armed = True
                    break
            if not armed:
                raise TimeoutError("Arm 확인 실패 (90s 내 armed 상태 미확인)")
            logger.info("[MAVLink] Armed ✓")

            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude,
            )
            t0 = time.time()
            while True:
                msg = drone.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=5)
                if msg and -msg.z >= altitude * 0.9:
                    break
                if time.time() - t0 > 60:
                    raise TimeoutError("이륙 고도 도달 실패 (60s 초과)")
            logger.info(f"[MAVLink] Takeoff 완료 → {altitude}m ({time.time() - t0:.1f}s)")
            return drone
        except Exception as e:
            last_err = e
            logger.warning(f"[MAVLink] 시도 {attempt}/{retries} 실패: {e}")
            time.sleep(3)

    raise RuntimeError(f"MAVLink 연결/이륙 {retries}회 모두 실패: {last_err}")


def launch_sitl_and_fly(logger, altitude):
    """SITL 실행 + MAVLink 연결/이륙. --no-mavproxy 실패 시 MAVProxy+--out 방식으로 대안 처리."""
    sitl_proc, sitl_logf = launch_sitl(logger, no_mavproxy=True)
    time.sleep(15)  # SITL 바이너리 부팅 대기
    try:
        drone = connect_and_takeoff(logger, altitude, MAVLINK_TCP_NO_MAVPROXY, retries=3)
        return drone, sitl_proc, sitl_logf
    except Exception as e:
        logger.warning(f"[SITL] --no-mavproxy(TCP 5760) 방식 실패 → MAVProxy+--out 방식으로 재시도: {e}")
        stop_sitl(logger, sitl_proc, sitl_logf)
        time.sleep(3)
        sitl_proc, sitl_logf = launch_sitl(logger, no_mavproxy=False)
        time.sleep(15)
        drone = connect_and_takeoff(logger, altitude, MAVLINK_UDP_MAVPROXY, retries=3)
        return drone, sitl_proc, sitl_logf


# ═══════════════════════════════════════════════════════════════════════
# [5] 실험 — 자동 벤치마크
# ═══════════════════════════════════════════════════════════════════════
def _capture_airsim_frame(ac_client):
    """AirSim에서 RGB 1프레임 캡처. (h, w, 3) BGR ndarray 반환."""
    import numpy as np
    import cosysairsim as airsim

    resp = ac_client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    r = resp[0]
    if r.width == 0:
        raise RuntimeError("AirSim 영상 캡처 실패 (width=0)")
    return np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)


def run_benchmark(logger, ac_client):
    """FPS를 두 지표로 분리 측정한다.
    (A) inference_fps — 메모리 이미지로 model() 호출만 반복. cuda.synchronize로
        비동기 완료를 보장하고 per-run 시간의 중앙값·표준편차 기록.
        AirSim IPC가 섞이면 모델 간 차이가 가려지므로 반드시 분리.
    (B) e2e_fps — 캡처+추론+후처리 전체. 실시간 운용 판정(5 FPS)은 이 값 기준."""
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    COMPARISON_DIR.mkdir(exist_ok=True)

    configs = [
        ("yolov8n_coco", BASE_DIR / "yolov8n.pt", 640),
        ("yolov8n_coco", BASE_DIR / "yolov8n.pt", 960),
        ("yolov8n_coco", BASE_DIR / "yolov8n.pt", 1280),
        ("yolov8s_visdrone", WEIGHTS_DIR / "yolov8s_visdrone.pt", 640),
        ("yolov8s_visdrone", WEIGHTS_DIR / "yolov8s_visdrone.pt", 960),
        ("yolov8s_visdrone", WEIGHTS_DIR / "yolov8s_visdrone.pt", 1280),
    ]

    use_cuda = torch.cuda.is_available()
    logger.info(f"[벤치마크] CUDA: {use_cuda}")

    logger.info("[벤치마크] AirSim 기준 영상 캡처 중...")
    img = _capture_airsim_frame(ac_client)
    cap_h, cap_w = img.shape[:2]
    logger.info(f"[벤치마크] 캡처 해상도: {cap_w}x{cap_h}")
    cv2.imwrite(str(COMPARISON_DIR / "capture_reference.png"), img)

    N_WARMUP, N_RUNS, N_E2E = 10, 100, 30
    rows = []
    for name, path, imgsz in configs:
        logger.info(f"[벤치마크] {name} @ {imgsz} 로드 중... ({path})")
        model = YOLO(str(path))

        # ── (A) 순수 추론: 메모리 이미지, per-run 타이밍 ──
        for _ in range(N_WARMUP):
            model(img, verbose=False, imgsz=imgsz)
        if use_cuda:
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            results = model(img, verbose=False, imgsz=imgsz)
            if use_cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median(times_ms))
        std_ms = float(np.std(times_ms))
        inf_fps = 1000.0 / med_ms

        # ── (B) 종단간: 캡처+추론+후처리 ──
        e2e_times = []
        for _ in range(N_E2E):
            t0 = time.perf_counter()
            frame = _capture_airsim_frame(ac_client)
            results = model(frame, verbose=False, imgsz=imgsz)
            _ = [(int(b.cls[0]), float(b.conf[0])) for b in results[0].boxes]  # 후처리
            if use_cuda:
                torch.cuda.synchronize()
            e2e_times.append(time.perf_counter() - t0)
        e2e_fps = 1.0 / float(np.median(e2e_times))

        n_det = len(results[0].boxes)
        passed = e2e_fps >= FPS_TARGET  # 판정은 종단간 기준

        annotated = results[0].plot()
        out_img = COMPARISON_DIR / f"{name}_{imgsz}.png"
        cv2.imwrite(str(out_img), annotated)

        logger.info(f"[벤치마크] {name}@{imgsz}: 추론 {inf_fps:.1f} FPS "
                    f"(중앙값 {med_ms:.1f}ms ± {std_ms:.1f}ms) | 종단간 {e2e_fps:.1f} FPS | "
                    f"탐지(참고) {n_det}개 | {'통과 ✓' if passed else '미달 ✗'}")
        rows.append({"model": name, "imgsz": imgsz,
                     "inference_fps": round(inf_fps, 2),
                     "inference_ms_median": round(med_ms, 2),
                     "inference_ms_std": round(std_ms, 2),
                     "e2e_fps": round(e2e_fps, 2),
                     # Blocks 맵에는 사람/차량이 없어 이 값은 오탐 포함 참고값임
                     # (예: COCO 모델이 큐브를 oven으로 오탐). 성능 지표는 model_comparison.csv 사용.
                     "detections_ref_not_metric": n_det, "pass_5fps_e2e": passed})

    csv_path = BENCHMARK_CSV
    try:
        f = open(csv_path, "w", newline="", encoding="utf-8-sig")
    except PermissionError:
        # 파일이 Excel 등에서 열려 있으면 잠긴다 — 측정을 날리지 말고 대체 이름으로 저장
        csv_path = BASE_DIR / "metrics" / f"benchmark_results_{dt.datetime.now():%H%M%S}.csv"
        logger.warning(f"[벤치마크] {BENCHMARK_CSV.name} 잠김(다른 프로그램에서 열림?) → {csv_path.name}으로 저장")
        f = open(csv_path, "w", newline="", encoding="utf-8-sig")
    with f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "imgsz", "inference_fps", "inference_ms_median",
            "inference_ms_std", "e2e_fps", "detections_ref_not_metric", "pass_5fps_e2e"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[벤치마크] 결과 저장: {csv_path} (캡처 {cap_w}x{cap_h})")

    logger.info("")
    logger.info("=== 벤치마크 비교 표 ===")
    logger.info(f"{'모델':18} {'imgsz':>6} {'추론FPS':>9} {'ms(중앙)':>9} {'±std':>7} {'종단간FPS':>10} {'판정':>5}")
    for row in rows:
        verdict = "통과" if row["pass_5fps_e2e"] else "미달"
        logger.info(f"{row['model']:18} {row['imgsz']:>6} {row['inference_fps']:>9.1f} "
                    f"{row['inference_ms_median']:>9.1f} {row['inference_ms_std']:>7.1f} "
                    f"{row['e2e_fps']:>10.1f} {verdict:>5}")

    # ── 측정 타당성 검증 (물리적으로 성립해야 하는 조건) ──
    by_key = {(row["model"], row["imgsz"]): row["inference_fps"] for row in rows}
    checks = [
        ("yolov8n@640 > yolov8s@640",
         by_key[("yolov8n_coco", 640)] > by_key[("yolov8s_visdrone", 640)]),
        ("yolov8n: 640 > 960 > 1280",
         by_key[("yolov8n_coco", 640)] > by_key[("yolov8n_coco", 960)]
         > by_key[("yolov8n_coco", 1280)]),
        ("yolov8s: 640 > 960 > 1280",
         by_key[("yolov8s_visdrone", 640)] > by_key[("yolov8s_visdrone", 960)]
         > by_key[("yolov8s_visdrone", 1280)]),
    ]
    all_ok = True
    for desc, ok in checks:
        logger.info(f"[검증] {desc}: {'성립 ✓' if ok else '위배 ✗'}")
        all_ok = all_ok and ok
    if not all_ok:
        logger.warning("[검증] 측정 타당성 조건 위배 — 결과를 신뢰하지 말고 원인 분석 필요")

    fail = [row for row in rows if not row["pass_5fps_e2e"]]
    if fail:
        logger.warning("[벤치마크] 종단간 5FPS 미달 구성:")
        for row in fail:
            logger.warning(f"  - {row['model']}@{row['imgsz']}: {row['e2e_fps']:.1f} FPS")
    else:
        logger.info("[벤치마크] 모든 구성이 종단간 5FPS 이상 통과 ✓")

    return {"rows": rows, "capture": (cap_w, cap_h), "validation_ok": all_ok}


# ═══════════════════════════════════════════════════════════════════════
# [6] 안전 종료
# ═══════════════════════════════════════════════════════════════════════
def safe_shutdown(logger, drone, unreal_proc, sitl_proc, sitl_logf, no_cleanup=False):
    if no_cleanup:
        logger.info("--no-cleanup 지정 — 프로세스 종료 생략 (디버깅 모드, 수동으로 정리할 것)")
        return

    if drone is not None:
        try:
            from pymavlink import mavutil
            logger.info("RTL(귀환) 명령 전송...")
            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            time.sleep(10)
            logger.info("Disarm...")
            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 21196, 0, 0, 0, 0, 0,
            )
            time.sleep(1)
        except Exception as e:
            logger.warning(f"RTL/Disarm 중 오류(무시하고 계속 종료): {e}")

    # 반드시 언리얼을 먼저 종료 → 그 다음 SITL/WSL 종료 (순서 바뀌면 언리얼이 응답 없음 상태가 됨)
    if unreal_proc is not None:
        try:
            logger.info("언리얼 프로세스 종료...")
            subprocess.run(["taskkill", "/F", "/IM", "UnrealEditor.exe"], capture_output=True)
        except Exception as e:
            logger.warning(f"언리얼 종료 중 오류(무시): {e}")
    time.sleep(2)

    if sitl_proc is not None:
        logger.info("SITL/WSL 프로세스 종료...")
        stop_sitl(logger, sitl_proc, sitl_logf)

    logger.info("안전 종료 절차 완료")


# ═══════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="드론 캡스톤 실험 전자동 오케스트레이터")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="YOLO 모델/해상도 자동 벤치마크")
    mode.add_argument("--patrol", action="store_true", help="사각형 경로 정찰 (patrol_detect)")
    mode.add_argument("--autonomous", action="store_true", help="NBV 기반 자율 정찰 (autonomous_patrol)")
    mode.add_argument("--collect", type=int, metavar="N", help="AirSim 데이터셋 N회 수집 (collect_data)")
    mode.add_argument("--synth", type=int, metavar="N",
                      help="합성 학습데이터 N장 수집+자동라벨 (ComputerVision 모드, SITL 불필요)")
    parser.add_argument("--alt", default="",
                         help="--synth 수집 고도(m) 쉼표 구분. 예: --alt 20  또는  --alt 15,20,25")
    parser.add_argument("--editor", action="store_true",
                         help="언리얼 에디터 수동 모드 (Play 직접 클릭, 데모 녹화용)")
    parser.add_argument("--no-cleanup", action="store_true",
                         help="종료 시 프로세스 정리 생략 (디버깅용)")
    args = parser.parse_args()

    logger, log_path = setup_logger()
    logger.info(f"=== run_experiment.py 시작 (로그: {log_path}) ===")

    run_t0 = time.time()
    drone = None
    unreal_proc = None
    sitl_proc = None
    sitl_logf = None
    success = False
    synth_mode = args.synth is not None

    try:
        with StepTimer(logger, "[1] 사전 점검"):
            preflight(logger)

        with StepTimer(logger, "[2] 언리얼 실행"):
            settings_override = (BASE_DIR / "configs" / "settings_synth.json") if synth_mode else None
            unreal_proc = launch_unreal(logger, editor_mode=args.editor,
                                        settings_override=settings_override)

        with StepTimer(logger, "[2] AirSim 연결 대기"):
            ac_client = wait_airsim_ready(logger)

        if not synth_mode:
            altitude = CRUISE_ALT
            if args.patrol:
                import patrol_detect
                altitude = -patrol_detect.WAYPOINTS[0][2]

            with StepTimer(logger, "[3+4] SITL 실행 + MAVLink 연결/이륙"):
                drone, sitl_proc, sitl_logf = launch_sitl_and_fly(logger, altitude)
        else:
            logger.info("[synth] ComputerVision 모드 — SITL/비행 생략, 텔레포트 수집")

        with StepTimer(logger, "[5] 실험 실행"):
            if args.benchmark:
                run_benchmark(logger, ac_client)
            elif args.patrol:
                import patrol_detect
                patrol_detect.run_patrol(drone, ac_client)
            elif args.autonomous:
                import autonomous_patrol
                autonomous_patrol.run_autonomous(drone, ac_client)
            elif args.collect is not None:
                import collect_data
                collect_data.run_collect(loop=args.collect)
            elif synth_mode:
                import collect_data
                alts = [float(a) for a in args.alt.split(",")] if args.alt else None
                collect_data.run_synth_collect(count=args.synth, log=logger.info,
                                               altitudes=alts)
                logger.info("[synth] 자동 라벨링 시작...")
                import auto_label
                # 대량 수집 시 검수 이미지 20장 (작업 5 요구사항)
                sys.argv = ["auto_label.py", "--qc", "20"]
                auto_label.main()

        success = True

    except Exception as e:
        logger.error(f"[실패] {e}", exc_info=True)
    finally:
        with StepTimer(logger, "[6] 안전 종료"):
            safe_shutdown(logger, drone, unreal_proc, sitl_proc, sitl_logf, no_cleanup=args.no_cleanup)

    total = time.time() - run_t0
    logger.info(f"=== 종료 (성공={success}, 총 소요 {total:.1f}s) ===")
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
