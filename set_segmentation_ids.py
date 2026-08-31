"""
set_segmentation_ids.py — Cosys-AirSim 인스턴스 segmentation 색상 매핑 구축 (작업 2)

중요: Cosys-AirSim은 레거시 AirSim의 "ID 부여" 방식(simSetSegmentationObjectID)이
아니라 **인스턴스 segmentation**을 사용한다. 모든 오브젝트가 자동으로 고유 색을
받으며, 색상은 colormap.npy[인스턴스 인덱스] (RGB 순서)로 결정된다.
(실측 확인: simSetSegmentationObjectID(".*", 0, True) → False 반환, 효과 없음)

따라서 여기서는 "ID를 설정"하는 대신
  1) simListInstanceSegmentationObjects() 로 오브젝트 목록을 얻고
  2) 사람/차량에 해당하는 인덱스를 이름 패턴으로 골라내
  3) colormap 에서 각 인스턴스의 정확한 색을 조회해
seg_color_map.json 에 기록한다.

인스턴스별 색이 다르므로 개체가 붙어 있어도 분리 라벨링이 가능하다(레거시 방식보다 유리).

실행: python set_segmentation_ids.py
"""
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cosysairsim as airsim
from cosysairsim.utils import load_colormap

BASE_DIR = Path(__file__).resolve().parent
COLOR_MAP_JSON = BASE_DIR / "configs" / "seg_color_map.json"

# place_persons.py 가 배치한 마네킹은 런타임에 SkeletalMeshActor_N 으로 나타난다
# (에디터 라벨 Person_* 은 런타임 오브젝트명으로 전달되지 않음).
PERSON_NAME_RE = re.compile(r"^SkeletalMeshActor", re.IGNORECASE)
VEHICLE_NAME_RE = re.compile(r"boxcar|vehicle", re.IGNORECASE)

PERSON_CLASS_ID = 0   # YOLO class id
VEHICLE_CLASS_ID = 1


def build_color_map(client, log=print):
    """인스턴스 목록 → 사람/차량 색상(RGB) 매핑 구축 후 JSON 저장."""
    objs = client.simListInstanceSegmentationObjects()
    cmap = load_colormap()
    log(f"[seg] 씬 인스턴스 총 {len(objs)}개")

    person = [(i, n) for i, n in enumerate(objs) if PERSON_NAME_RE.search(n)]
    vehicle = [(i, n) for i, n in enumerate(objs) if VEHICLE_NAME_RE.search(n)]

    log(f"[seg] 사람 인스턴스 {len(person)}개: {[n for _, n in person]}")
    log(f"[seg] 차량 인스턴스 {len(vehicle)}개: {[n for _, n in vehicle]}")
    if not person:
        raise RuntimeError(
            "사람 인스턴스를 찾지 못했습니다. Blocks 맵에 마네킹이 배치되어 있는지 확인하세요 "
            "(place_persons.py 실행 필요).")

    def entry(items, class_id):
        return [{"name": n, "instance_index": int(i),
                 "rgb": [int(v) for v in cmap[i]], "class_id": class_id}
                for i, n in items]

    # 사람 위치(NED)도 함께 기록 — 수집 시 상공 촬영 지점으로 사용
    poses = client.simListInstanceSegmentationPoses(ned=True)
    person_entries = entry(person, PERSON_CLASS_ID)
    for e in person_entries:
        p = poses[e["instance_index"]].position
        e["ned"] = [float(p.x_val), float(p.y_val), float(p.z_val)]

    color_map = {
        "note": "Cosys-AirSim 인스턴스 segmentation. rgb는 colormap.npy[instance_index] 값이며 "
                "seg 이미지 바이트 순서와 동일(RGB). cv2.imwrite/imread 왕복은 배열을 보존하므로 "
                "auto_label.py 는 이 값을 그대로 비교한다.",
        "classes": {str(PERSON_CLASS_ID): "person", str(VEHICLE_CLASS_ID): "vehicle"},
        "person": person_entries,
        "vehicle": entry(vehicle, VEHICLE_CLASS_ID),
    }
    with open(COLOR_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(color_map, f, indent=2, ensure_ascii=False)
    log(f"[seg] 매핑 저장: {COLOR_MAP_JSON} "
        f"(사람 {len(person_entries)}, 차량 {len(color_map['vehicle'])})")
    return color_map


# 하위 호환 별칭 (기존 호출부 유지)
setup_segmentation = build_color_map


def main():
    client = airsim.VehicleClient()
    client.confirmConnection()
    cm = build_color_map(client)
    print("\n사람 인스턴스 색상(RGB) 및 NED 위치:")
    for e in cm["person"]:
        print(f"  {e['name']}: rgb={e['rgb']} ned=({e['ned'][0]:.2f}, {e['ned'][1]:.2f}, {e['ned'][2]:.2f})")


if __name__ == "__main__":
    main()
