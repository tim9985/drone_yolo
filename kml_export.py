"""
kml_export.py — 탐지 결과를 지도에 찍히는 형태(KML)로 내보내기

왜 필요한가
  구조대에 "화면 좌표 (717, 643)에 사람이 있다"는 쓸모가 없다. 위경도가 나와야 하고,
  그것도 숫자 목록이 아니라 **지도에 찍힌 점**이어야 현장에서 쓸 수 있다.
  KML 은 Google Earth · QGIS · 국토정보플랫폼에서 그대로 열린다.

프레임 단위 탐지 → 사람 단위 신고
  드론이 한 사람 위를 5초 지나가면 탐지가 50번 찍힌다. 그대로 내보내면 지도에 핀이
  50개 꽂혀 아무도 못 읽는다. 그래서 **같은 대상을 하나로 묶는다.**

    1) track_id 가 같으면 같은 대상
    2) track_id 가 끊겨도 실좌표가 merge_radius(기본 3m) 안이면 같은 대상
    3) 묶인 위치는 **중앙값** — 평균은 튀는 값 하나에 끌려간다

  묶고 나면 "몇 번 관측됐는가"가 자연히 증거 강도가 된다. 바위 그림자는 각도가
  바뀌면 사라지므로 무리를 이루지 못하고, 진짜 사람은 여러 프레임에 걸쳐 쌓인다.
  → **min_hits 미만인 무리는 '미확인 후보'로 내려보내 기본 숨김 처리**한다.
    (프레임 단위로 같은 판정을 걸었을 때는 진짜 사람까지 막혔다. 시간·공간 증거가
     쌓인 뒤에 거르는 이 위치가 맞다.)

사용
  from kml_export import DetectionCollector
  col = DetectionCollector(origin_gps=(35.1796, 129.0756), origin_alt_m=30.0)
  col.add_drone(t, north, east, alt)                       # 매 프레임
  col.add(t, "person", 0.62, north, east, alt, track_id=3) # 탐지마다
  col.write_kml("results/detections.kml")

단독 실행: python kml_export.py --selftest
"""
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from coord_transform import ned_to_gps

DEFAULT_ORIGIN = (35.1796, 129.0756)   # settings.json 의 OriginGeopoint 와 일치시킬 것
DEFAULT_ORIGIN_ALT = 30.0

MERGE_RADIUS_M = 3.0     # 이 거리 안이면 같은 대상으로 본다
MIN_HITS = 3             # 이 횟수 미만이면 '미확인 후보'


def read_origin_from_settings(path=r"C:\AirSim\settings.json"):
    """AirSim settings.json 의 OriginGeopoint 를 읽는다.

    이 값이 시뮬레이터의 (0,0,0) 이 실제 지구상 어디인지를 정한다. 여기가 틀리면
    KML 의 모든 핀이 통째로 엉뚱한 곳에 찍히므로 하드코딩하지 않고 설정에서 읽는다.
    실기체로 옮길 때는 이 함수 대신 이륙 지점의 GPS(getHomeGeoPoint)를 쓰면 된다.
    """
    import json
    try:
        g = json.loads(Path(path).read_text(encoding="utf-8")).get("OriginGeopoint", {})
        lat, lon = float(g["Latitude"]), float(g["Longitude"])
        return (lat, lon), float(g.get("Altitude", DEFAULT_ORIGIN_ALT))
    except Exception as e:
        print(f"[KML] settings.json 원점을 못 읽어 기본값 사용 ({e})")
        return DEFAULT_ORIGIN, DEFAULT_ORIGIN_ALT


def _esc(s):
    """KML 은 XML 이므로 &, <, > 를 반드시 이스케이프해야 한다."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


@dataclass
class Target:
    """같은 대상으로 묶인 탐지들"""
    cls_name: str
    xs: list = field(default_factory=list)     # NED north (m)
    ys: list = field(default_factory=list)     # NED east (m)
    zs: list = field(default_factory=list)     # 지면 기준 높이 (m)
    confs: list = field(default_factory=list)
    times: list = field(default_factory=list)
    track_ids: set = field(default_factory=set)

    @property
    def hits(self):
        return len(self.xs)

    @property
    def pos(self):
        """중앙값 위치 — 튀는 값에 끌리지 않는다"""
        return _median(self.xs), _median(self.ys), _median(self.zs)

    @property
    def conf_max(self):
        return max(self.confs) if self.confs else 0.0

    @property
    def conf_mean(self):
        return sum(self.confs) / len(self.confs) if self.confs else 0.0

    @property
    def span(self):
        return (min(self.times), max(self.times)) if self.times else (0.0, 0.0)

    def dist_to(self, x, y):
        px, py, _ = self.pos
        return math.hypot(px - x, py - y)


class DetectionCollector:
    """탐지를 모아 대상 단위로 병합하고 KML/CSV 로 내보낸다."""

    def __init__(self, origin_gps=DEFAULT_ORIGIN, origin_alt_m=DEFAULT_ORIGIN_ALT,
                 merge_radius_m=MERGE_RADIUS_M, min_hits=MIN_HITS):
        self.origin = tuple(origin_gps)
        self.origin_alt = float(origin_alt_m)
        self.merge_radius = float(merge_radius_m)
        self.min_hits = int(min_hits)
        self.targets = []
        self.raw = []             # 병합 전 원본 탐지. 반경을 바꿔 재분석하려면 이게 있어야 한다
        self.track_path = []      # 드론 궤적 [(t, north, east, alt)]
        self.t0 = None

    # ── 입력 ──────────────────────────────────────────────
    def add_drone(self, t, north, east, alt_m):
        if self.t0 is None:
            self.t0 = t
        self.track_path.append((t, float(north), float(east), float(alt_m)))

    def add(self, t, cls_name, conf, north, east, height_m=0.0, track_id=None,
            roll_deg=None, pitch_deg=None):
        """탐지 1건 추가. north/east 는 홈 기준 NED(m), height_m 은 지면 위 높이."""
        if math.isfinite(north) and math.isfinite(east):
            self.raw.append((float(t), str(cls_name), float(conf), float(north),
                             float(east), float(height_m), track_id,
                             roll_deg, pitch_deg))
        if not (math.isfinite(north) and math.isfinite(east)):
            return
        if self.t0 is None:
            self.t0 = t
        tgt = self._match(cls_name, north, east, track_id)
        if tgt is None:
            tgt = Target(cls_name=cls_name)
            self.targets.append(tgt)
        tgt.xs.append(float(north)); tgt.ys.append(float(east))
        tgt.zs.append(float(height_m)); tgt.confs.append(float(conf))
        tgt.times.append(float(t))
        if track_id is not None:
            tgt.track_ids.add(track_id)

    def _match(self, cls_name, north, east, track_id):
        """1) track_id 일치 → 2) 거리 근접. 둘 다 없으면 새 대상."""
        if track_id is not None:
            for t in self.targets:
                if t.cls_name == cls_name and track_id in t.track_ids:
                    return t
        best, best_d = None, self.merge_radius
        for t in self.targets:
            if t.cls_name != cls_name:
                continue
            d = t.dist_to(north, east)
            if d <= best_d:
                best, best_d = t, d
        return best

    # ── 출력 ──────────────────────────────────────────────
    def confirmed(self):
        return sorted([t for t in self.targets if t.hits >= self.min_hits],
                      key=lambda t: -t.hits)

    def candidates(self):
        return sorted([t for t in self.targets if t.hits < self.min_hits],
                      key=lambda t: -t.hits)

    def _gps(self, north, east):
        return ned_to_gps(self.origin, north, east)

    def summary(self):
        c, k = self.confirmed(), self.candidates()
        return {"targets_confirmed": len(c), "targets_candidate": len(k),
                "detections_total": sum(t.hits for t in self.targets),
                "path_points": len(self.track_path)}

    def write_kml(self, path, title="자율 정찰 드론 — 탐지 결과", note=""):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        conf_list, cand_list = self.confirmed(), self.candidates()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")

        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<kml xmlns="http://www.opengis.net/kml/2.2">', '<Document>',
               f'<name>{_esc(title)}</name>',
               '<description><![CDATA['
               f'생성 {stamp}<br/>확인된 대상 {len(conf_list)}명 · '
               f'미확인 후보 {len(cand_list)}건<br/>{_esc(note)}]]></description>']

        # 스타일 — 증거가 많을수록 눈에 띄게
        for sid, color, scale in [("strong", "ff0000ff", 1.3),   # 빨강 (aabbggrr)
                                  ("medium", "ff00a5ff", 1.1),   # 주황
                                  ("weak",   "ff00ffff", 0.9)]:  # 노랑
            out += [f'<Style id="t_{sid}"><IconStyle><color>{color}</color>'
                    f'<scale>{scale}</scale><Icon><href>'
                    'http://maps.google.com/mapfiles/kml/shapes/target.png'
                    '</href></Icon></IconStyle>'
                    '<LabelStyle><scale>0.9</scale></LabelStyle></Style>']
        out += ['<Style id="cand"><IconStyle><color>96b4b4b4</color><scale>0.7</scale>'
                '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/open-diamond.png'
                '</href></Icon></IconStyle><LabelStyle><scale>0.7</scale></LabelStyle></Style>',
                '<Style id="path"><LineStyle><color>ffd6781b</color><width>3</width>'
                '</LineStyle></Style>']

        def placemark(idx, t, style, prefix):
            n, e, h = t.pos
            lat, lon = self._gps(n, e)
            t_start, t_end = t.span
            rel = (t_start - self.t0) if self.t0 is not None else t_start
            desc = (f'<![CDATA['
                    f'<b>신뢰도</b> 최고 {t.conf_max:.2f} · 평균 {t.conf_mean:.2f}<br/>'
                    f'<b>관측</b> {t.hits}회 ({t_end - t_start:.1f}초 동안)<br/>'
                    f'<b>최초 관측</b> 시작 후 {rel:.1f}초<br/>'
                    f'<b>좌표</b> {lat:.7f}, {lon:.7f}<br/>'
                    f'<b>홈 기준</b> 북 {n:+.1f}m · 동 {e:+.1f}m'
                    f']]>')
            return ('<Placemark>'
                    f'<name>{_esc(prefix)} {idx}</name>'
                    f'<description>{desc}</description>'
                    f'<styleUrl>#{style}</styleUrl>'
                    '<Point><altitudeMode>clampToGround</altitudeMode>'
                    f'<coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point>'
                    '</Placemark>')

        out += ['<Folder><name>확인된 대상 (%d)</name><open>1</open>' % len(conf_list)]
        for i, t in enumerate(conf_list, 1):
            style = "t_strong" if t.hits >= 10 else ("t_medium" if t.hits >= 5 else "t_weak")
            out.append(placemark(i, t, style, "대상"))
        out.append('</Folder>')

        # 미확인 후보는 기본 숨김 — 지도를 어지럽히지 않으면서 확인은 가능하게
        if cand_list:
            out += ['<Folder><name>미확인 후보 (%d)</name><visibility>0</visibility>'
                    '<open>0</open>' % len(cand_list)]
            for i, t in enumerate(cand_list, 1):
                out.append(placemark(i, t, "cand", "후보"))
            out.append('</Folder>')

        if len(self.track_path) >= 2:
            coords = []
            for _, n, e, alt in self.track_path:
                lat, lon = self._gps(n, e)
                coords.append(f"{lon:.8f},{lat:.8f},{max(alt, 0.0):.1f}")
            out += ['<Folder><name>비행 경로</name>',
                    '<Placemark><name>탐색 궤적</name><styleUrl>#path</styleUrl>',
                    '<LineString><tessellate>1</tessellate>',
                    '<altitudeMode>relativeToGround</altitudeMode>',
                    '<coordinates>' + " ".join(coords) + '</coordinates>',
                    '</LineString></Placemark></Folder>']

        out += ['</Document>', '</kml>']
        p.write_text("\n".join(out), encoding="utf-8")
        return p


# ── 자체 검증 ──────────────────────────────────────────────
def _selftest():
    import xml.etree.ElementTree as ET
    ok = True

    col = DetectionCollector(min_hits=3, merge_radius_m=3.0)
    # 사람 A: 20프레임에 걸쳐 관측, 위치가 ±0.5m 흔들림
    for i in range(20):
        col.add(100.0 + i * 0.1, "person", 0.6,
                10.0 + (0.5 if i % 2 else -0.5), 5.0, 0.0, track_id=1)
    # 사람 B: track_id 가 중간에 끊겼지만 같은 자리 → 하나로 묶여야 한다
    for i in range(6):
        col.add(120.0 + i * 0.1, "person", 0.5, -8.0, 12.0, 0.0, track_id=2)
    for i in range(6):
        col.add(130.0 + i * 0.1, "person", 0.5, -8.4, 12.3, 0.0, track_id=7)
    # 순간 오탐 2건: 서로 멀고 각 1회
    col.add(140.0, "person", 0.3, 30.0, -20.0, 0.0, track_id=11)
    col.add(141.0, "person", 0.3, -25.0, 28.0, 0.0, track_id=12)
    for i in range(5):
        col.add_drone(100.0 + i, i * 3.0, 0.0, 20.0)

    conf, cand = col.confirmed(), col.candidates()
    print(f"  확인 대상 {len(conf)}명 / 미확인 후보 {len(cand)}건")
    if len(conf) != 2:
        print(f"  [실패] 확인 대상이 2명이어야 하는데 {len(conf)}명"); ok = False
    if len(cand) != 2:
        print(f"  [실패] 후보가 2건이어야 하는데 {len(cand)}건"); ok = False
    # track_id 가 끊긴 B 가 하나로 묶였는지
    b = [t for t in conf if t.hits == 12]
    if not b:
        print("  [실패] track_id 가 끊긴 탐지들이 하나로 병합되지 않음"); ok = False
    else:
        n, e, _ = b[0].pos
        if not (-8.5 <= n <= -7.9 and 11.9 <= e <= 12.4):
            print(f"  [실패] 병합 위치가 이상함: 북{n:.2f} 동{e:.2f}"); ok = False
        else:
            print(f"  track_id 끊김 병합 OK — 12회 관측, 북{n:.2f} 동{e:.2f}")
    # 중앙값이 흔들림에 강한지 (A 는 ±0.5 진동 → 중앙값이 10 근처)
    a = [t for t in conf if t.hits == 20]
    if a:
        n, _, _ = a[0].pos
        print(f"  진동 ±0.5m 에서 추정 위치 북{n:.2f} (참값 10.00)")
        if abs(n - 10.0) > 0.6:
            print("  [실패] 중앙값 추정이 부정확"); ok = False

    out = Path("results/_kml_selftest.kml")
    col.write_kml(out, note="자체 검증용")
    try:
        root = ET.parse(out).getroot()
    except ET.ParseError as e:
        print(f"  [실패] KML 이 유효한 XML 이 아님: {e}"); ok = False
        return ok
    ns = "{http://www.opengis.net/kml/2.2}"
    pms = root.iter(f"{ns}Placemark")
    npm = sum(1 for _ in pms)
    print(f"  XML 파싱 OK — Placemark {npm}개 (대상 2 + 후보 2 + 궤적 1 = 5 기대)")
    if npm != 5:
        print("  [실패] Placemark 수 불일치"); ok = False

    # 좌표가 부산 근방인지 (원점 35.1796, 129.0756)
    for c in root.iter(f"{ns}coordinates"):
        lon, lat = [float(v) for v in c.text.strip().split()[0].split(",")[:2]]
        if not (128.9 < lon < 129.3 and 35.0 < lat < 35.4):
            print(f"  [실패] 좌표가 범위를 벗어남: {lat}, {lon}"); ok = False
            break

    # XML 이스케이프 확인
    col2 = DetectionCollector()
    col2.add(0.0, "person", 0.5, 1.0, 1.0, 0.0, track_id=1)
    p2 = col2.write_kml("results/_kml_esc.kml", title='위험 & <시험> "따옴표"')
    try:
        ET.parse(p2)
        print("  특수문자 이스케이프 OK")
    except ET.ParseError as e:
        print(f"  [실패] 이스케이프 누락: {e}"); ok = False
    for f in (out, p2):
        try: Path(f).unlink()
        except Exception: pass
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("kml_export 자체 검증")
        sys.exit(0 if _selftest() else 1)
    print(__doc__)


def _write_raw_csv(self, path):
    """병합 전 원본 탐지를 CSV 로 남긴다.

    비행을 다시 하지 않고도 병합 반경·최소 관측횟수를 바꿔 가며
    재분석할 수 있어야 한다 (merge_tune.py 가 이 파일을 읽는다)."""
    import csv
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "class", "conf", "north_m", "east_m", "height_m", "track_id",
                    "roll_deg", "pitch_deg"])
        for r in self.raw:
            w.writerow([f"{r[0]:.3f}", r[1], f"{r[2]:.4f}",
                        f"{r[3]:.3f}", f"{r[4]:.3f}", f"{r[5]:.3f}",
                        "" if r[6] is None else r[6],
                        "" if r[7] is None else f"{r[7]:.2f}",
                        "" if r[8] is None else f"{r[8]:.2f}"])
    return str(p)


DetectionCollector.write_raw_csv = _write_raw_csv
