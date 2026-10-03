#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""下载卫星切片拼成一张大图，并在图上标点。

拼好的图旁边会写一个同名 .json（缩放级别、左上角切片号、中心点），
evidence.py 和 mark 子命令靠它把经纬度换成图上像素。

## 数据源（--source）

| 名 | 服务 | 坐标系 | 国内直连 | 备注 |
|---|---|---|---|---|
| `amap`（**国内默认**） | 高德卫星 `webst0{1-4}.is.autonavi.com` | GCJ-02 | 是 | 免key。国内影像最清晰、最新 |
| `tianditu` | 天地图卫星（需在 SKILL_TIANDITU_KEY 放 key） | WGS84(CGCS2000近似) | 是 | key申请见references/data-sources.md |
| `google` | Google 卫星 `mt1.google.com` | WGS84 | **需代理** | 影像最锐利，但要代理 |
| `esri` | Esri World Imagery | WGS84 | 是 | 备用，国内部分地区较旧 |

**国内默认走 `amap`**，不需要任何代理。`--source auto`（等价于不写--source 时）
会先试高德；高德失败再试天地图；都没有才回落 Google/Esri。

**GCJ-02 说明（实测结论，别自己再纠偏）**：高德 `appmaptile` 接口的**瓦片编号口径就是
WGS84**——实测把 WGS84 坐标纠偏成 GCJ-02 再算瓦片号，会落到空白区（用路网瓦片验证：
纠偏后 24.6 KB 空白，正确号192 B 有路网）。所以本脚本对所有源都用 WGS84 算瓦片号，
`meta` 里 `tile_crs` 恒为 `wgs84`。**调用方一律传 WGS84**，和坐标盘点算口径一致。
（注意这和`geo.py convert` 给人看的 GCJ-02 坐标不是一回事：坐标展示要纠偏，
瓦片编号不要纠偏。）

## 代理

`--proxy` 或环境变量 `GEO_PROXY`（例：`socks5h://127.0.0.1:10808`）。
国内源不需要代理；只有 `google`/`esri` 海外段、Overpass、Google 街景需要。
**先跑 `netcheck.py` 看哪些源当前真的通**，别默认代理是配好的。

示例：
  tiles.py fetch 22.6050,114.0540 --zoom 18 --radius 4 --out area.jpg
  tiles.py mark area.jpg --points panos.json --out area_panos.jpg --label
  tiles.py mark area.jpg --points cands.json --geojson power.geojson --geojson rail.geojson \
           --sector 22.6045,114.0520,265,40,3000 --out area_lines.jpg       # 叠线状要素 + 机位视野扇形
  tiles.py sheet --points cands.json --zoom 18 --out cands_sheet.jpg     # 候选点逐个居中出缩略图，带编号
  tiles.py px2ll area.jpg --px 812,440 --px 300,95                      # 拼图像素 → 经纬度（裁剪图加 --crop x0,y0 --scale s）
  tiles.py sheet --grid <s,w,n,e> --zoom 17 --size 320 --cols 5 --out town.jpg   # 整片城区铺网格逐格看（找操场、厂房）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import geo  # noqa: E402

SOURCES = {
    "amap": "https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    "tianditu": "https://t{s}.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
                "&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={z}"
                "&TILEROW={y}&TILECOL={x}&tk={tk}",
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

# 用 GCJ-02 火星坐标的源：注意高德 appmaptile 接口实测接受 **WGS84 切片号**
# （已用路网瓦片验证：纠偏后的号会落到空白区），所以这里不做纠偏，
# 卫星图和路网瓦片编号口径一致。只有当某个源明确要求火星坐标时才加进这个集合。
GCJ_SOURCES: set[str] = set()
# 免key 直连优先级；google 放最后（要代理）
AUTO_ORDER = ["amap", "esri", "google"]


def _src_fmt(source: str, x: int, y: int, z: int) -> str:
    tpl = SOURCES[source]
    if source == "amap":
        # 高德按 x+y mod 4 分到 4 台服务器
        tpl = tpl.format(s=(x + y) % 4 + 1, x=x, y=y, z=z)
        return tpl
    if source == "tianditu":
        tk = os.environ.get("SKILL_TIANDITU_KEY", "")
        if not tk:
            return ""
        return tpl.format(s=(x + y) % 4 + 1, x=x, y=y, z=z, tk=tk)
    return tpl.format(x=x, y=y, z=z)


_PROBED: dict[str, bool] = {}


def _src_alive(src: str, proxy: str | None) -> bool:
    """探一个源通不通（拿一张已知有图的瓦片试，结果缓存）。"""
    if src in _PROBED:
        return _PROBED[src]
    url = _src_fmt(src, 99, 48, 7)
    ok = False
    if url:
        ok = _get(url, Path(".geo-cache") / f"probe_{src}.jpg", proxy, retries=0)
    _PROBED[src] = ok
    return ok


def _resolve_sheet_source(source: str, proxy: str | None) -> str:
    """auto → 探一遍，第一个通的源。sheet/mark 这类只要单张瓦片的路径用。"""
    if source != "auto":
        return source
    for cand in (["google"] if proxy else []) + AUTO_ORDER:
        if cand == "tianditu" and not os.environ.get("SKILL_TIANDITU_KEY"):
            continue
        if _src_alive(cand, proxy):
            return cand
    sys.exit("tiles：所有卫星源都不可用。跑 scripts/netcheck.py 诊断；"
             "要用 Google 就 --proxy socks5h://127.0.0.1:7890（示例端口）。")


def _get(url: str, path: Path, proxy: str | None, retries: int = 2) -> bool:
    if path.exists() and path.stat().st_size > 1000:
        return True
    for attempt in range(retries + 1):
        cmd = ["curl", "-s", "-m", "60", "-o", str(path), url]
        if proxy:
            cmd[1:1] = ["-x", proxy]
        subprocess.run(cmd, check=False)
        if path.exists() and path.stat().st_size > 1000:
            return True
    return False


def fetch(center: tuple[float, float], zoom: int, radius: int, out: Path, source: str,
          proxy: str | None, cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)

    def try_source(src: str) -> dict | None:
        """试一个源；成功返回 meta，失败返回 None。"""
        gx, gy = geo.ll2px(zoom, *center)
        cx, cy = int(gx // 256), int(gy // 256)
        xs = range(cx - radius, cx + radius + 1)
        ys = range(cy - radius, cy + radius + 1)
        jobs = [(x, y) for x in xs for y in ys]
        total = len(jobs)
        nproc = 8 if src in ("google", "tianditu") else 16

        def job(t):
            x, y = t
            p = cache / f"{src}_{zoom}_{x}_{y}.jpg"
            url = _src_fmt(src, x, y, zoom)
            if not url:
                return t, p, False
            ok = _get(url, p, proxy)
            return t, p, ok

        img = Image.new("RGB", (256 * len(xs), 256 * len(ys)), "black")
        failed = 0
        got = 0
        with ThreadPoolExecutor(nproc) as ex:
            for (x, y), p, ok in ex.map(job, jobs):
                if not ok:
                    failed += 1
                    continue
                try:
                    img.paste(Image.open(p), ((x - xs[0]) * 256, (y - ys[0]) * 256))
                    got += 1
                except Exception:
                    failed += 1
        # 一半以上失败就当这个源不可用
        if got == 0 or failed * 2 > total:
            return None
        img.save(out, quality=90)
        meta = {"zoom": zoom, "origin_tile": [xs[0], ys[0]], "center": list(center), "source": src,
                "input_crs": "wgs84", "tile_crs": "gcj02" if src in GCJ_SOURCES else "wgs84",
                "size": list(img.size), "m_per_px": geo.meters_per_px(zoom, center[0]),
                "failed_tiles": failed}
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    order = AUTO_ORDER if source in ("auto", "", None) else [source]
    if source == "auto" and proxy:
        # 配了代理就把Google 提到前面试（用户明确要代理，通常是想用 Google）
        order = ["google", "amap", "tianditu", "esri"]
    errs = []
    for src in order:
        if src == "tianditu" and not os.environ.get("SKILL_TIANDITU_KEY"):
            errs.append(f"{src}: 没设 SKILL_TIANDITU_KEY，跳过")
            continue
        m = try_source(src)
        if m:
            if errs:
                print(f"[tiles] 回落：{src}（前面的源试过不可用）", file=sys.stderr)
            return m
        errs.append(f"{src}: 切片下载失败（多半是网络不通或要代理）")
    raise SystemExit("tiles.py：所有卫星源都不可用。\n  " + "\n  ".join(errs) +
                     "\n国内源本应直连，先跑 scripts/netcheck.py 诊断；"
                     "要用 Google 就 --proxy socks5h://127.0.0.1:10808（示例端口）。")


class Mosaic:
    """经纬度 ↔ 拼图像素。

    外部一律用 WGS84（和坐标盘、geo.py 口径一致）。meta 里记了瓦片实际用的坐标系
    （高德/腾讯/百度是 GCJ-02），这里自动双向转换，调用方不用管。
    """

    def __init__(self, image: Path):
        self.meta = json.loads(Path(image).with_suffix(".json").read_text(encoding="utf-8"))
        self.z = self.meta["zoom"]
        self.ox, self.oy = self.meta["origin_tile"]
        # 瓦片坐标系 → WGS84 的逆转换
        self._gcj_tile = self.meta.get("tile_crs") == "gcj02" or \
            self.meta.get("source") in GCJ_SOURCES

    def to_px(self, lat: float, lon: float) -> tuple[float, float]:
        if self._gcj_tile:
            lat, lon = geo.wgs2gcj(lat, lon)
        x, y = geo.ll2px(self.z, lat, lon)
        return x - self.ox * 256, y - self.oy * 256

    def to_ll(self, x: float, y: float) -> tuple[float, float]:
        lat, lon = geo.px2ll(self.z, x + self.ox * 256, y + self.oy * 256)
        if self._gcj_tile:
            lat, lon = geo.gcj2wgs(lat, lon)
        return lat, lon


PALETTE = ["yellow", "cyan", "magenta", "lime", "orange", "white", "red", "deepskyblue"]


LINE_COLORS = ["orange", "deepskyblue", "magenta", "lime", "white", "red"]


def _draw_geojson(d: ImageDraw.ImageDraw, m: "Mosaic", gj: dict, color: str) -> int:
    """GeoJSON 的线、面、点画到拼图上（线宽 3，面只画轮廓）。返回画了几个要素。"""
    n = 0
    for ft in gj.get("features", []):
        g = ft.get("geometry") or {}
        t, cs = g.get("type"), g.get("coordinates")
        if not cs:
            continue
        lines = {"LineString": [cs], "MultiLineString": cs, "Polygon": cs,
                 "MultiPolygon": [ring for poly in cs for ring in poly]}.get(t)
        if lines is not None:
            for ln in lines:
                xy = [m.to_px(c[1], c[0]) for c in ln]
                if len(xy) >= 2:
                    d.line(xy, fill=color, width=3)
            n += 1
        elif t == "Point":
            x, y = m.to_px(cs[1], cs[0])
            d.rectangle([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
            n += 1
    return n


def _draw_sector(d: ImageDraw.ImageDraw, m: "Mosaic", spec: str) -> None:
    """lat,lon,朝向,水平视角,半径m → 机位视野扇形。"""
    lat, lon, hd, fov, rng = map(float, spec.split(","))
    pts = [m.to_px(lat, lon)]
    steps = max(4, int(fov // 3))
    for k in range(steps + 1):
        a = hd - fov / 2 + fov * k / steps
        pts.append(m.to_px(*geo.dest((lat, lon), a, rng)))
    d.line(pts + [pts[0]], fill="yellow", width=3)
    mid = m.to_px(*geo.dest((lat, lon), hd, rng))
    d.line([pts[0], mid], fill="yellow", width=1)
    x, y = pts[0]
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill="red", outline="yellow", width=2)


def mark(image: Path, points: dict, out: Path, label: bool, geojsons: list[Path] | None = None,
         sectors: list[str] | None = None) -> None:
    """points: {name: [lat, lon]}（wgs）。名字按第一个空格前的前缀分色；--label 时写名字的前 12 个字。"""
    m = Mosaic(image)
    img = Image.open(image).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, gp in enumerate(geojsons or []):
        n = _draw_geojson(d, m, json.loads(Path(gp).read_text(encoding="utf-8")), LINE_COLORS[i % len(LINE_COLORS)])
        print(f"{gp}: {n} 个要素，颜色 {LINE_COLORS[i % len(LINE_COLORS)]}")
    for sp in sectors or []:
        _draw_sector(d, m, sp)
    groups: dict[str, str] = {}
    for name, ll in points.items():
        lat, lon = ll[0], ll[1]
        prefix = str(name).split(" ")[0].rstrip("0123456789") or "_"
        color = groups.setdefault(prefix, PALETTE[len(groups) % len(PALETTE)])
        x, y = m.to_px(lat, lon)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline="black")
        if label:
            d.text((x + 7, y - 6), str(name)[:12], fill=color)
    img.save(out, quality=90)


def grid_points(bbox: str, zoom: int, size: int, step: float | None) -> dict:
    """s,w,n,e 铺成网格点；默认格距 = 每格覆盖宽度的九成。"""
    import math

    s, w, n, e = (float(v) for v in bbox.split(","))
    mpp = 156543.03392 * math.cos(math.radians((s + n) / 2)) / 2 ** zoom
    step = step or size * mpp * 0.9
    dlat, dlon = step / 110574, step / (111320 * math.cos(math.radians((s + n) / 2)))
    pts, r, lat = {}, 0, n - dlat / 2
    while lat > s:
        c, lon = 0, w + dlon / 2
        while lon < e:
            pts[f"r{r:02d}c{c:02d}"] = [round(lat, 6), round(lon, 6)]
            c, lon = c + 1, lon + dlon
        r, lat = r + 1, lat - dlat
    return pts


def sheet(points: dict, zoom: int, size: int, cols: int, out: Path, source: str, proxy: str | None, cache: Path) -> list[Path]:
    """每个候选点居中出一张 size×size 的卫星缩略图，带编号和名字，拼成一页（多了自动分页）。"""
    cache.mkdir(parents=True, exist_ok=True)
    source = _resolve_sheet_source(source, proxy)
    items = list(points.items())
    per = cols * cols
    pages = []
    for pi in range(0, len(items), per):
        chunk = items[pi:pi + per]
        rows = (len(chunk) + cols - 1) // cols
        S = Image.new("RGB", (cols * size, rows * size), "black")
        dr = ImageDraw.Draw(S)
        for k, (name, ll) in enumerate(chunk):
            gx, gy = geo.ll2px(zoom, ll[0], ll[1])
            x0, y0 = gx - size / 2, gy - size / 2
            tile = Image.new("RGB", (size, size), "gray")
            for tx in range(int(x0 // 256), int((x0 + size) // 256) + 1):
                for ty in range(int(y0 // 256), int((y0 + size) // 256) + 1):
                    p = cache / f"{source}_{zoom}_{tx}_{ty}.jpg"
                    if _get(_src_fmt(source, tx, ty, zoom), p, proxy):
                        try:
                            tile.paste(Image.open(p), (int(tx * 256 - x0), int(ty * 256 - y0)))
                        except Exception:  # noqa: BLE001
                            pass
            cx, cy = (k % cols) * size, (k // cols) * size
            S.paste(tile, (cx, cy))
            c = size / 2
            dr.line([cx + c - 12, cy + c, cx + c + 12, cy + c], fill="red", width=2)
            dr.line([cx + c, cy + c - 12, cx + c, cy + c + 12], fill="red", width=2)
            dr.rectangle([cx, cy, cx + size, cy + 20], fill="black")
            dr.text((cx + 4, cy + 3), f"#{pi + k + 1} {str(name)[:28]}", fill="yellow")
        o = out if pi == 0 else out.with_name(f"{out.stem}_{pi // per + 1}{out.suffix}")
        S.save(o, quality=88)
        pages.append(o)
    return pages



def _neg_coords(argv: list[str]) -> list[str]:
    """argparse 把 -1.45,-48.5 这种负坐标当成选项名；前面补个空格就当普通值（float 会忽略空格）。南半球、西半球的题都要用。"""
    return [" " + a if re.match(r"^-\d[\d.]*(,-?[\d.]+)+$", a) else a for a in argv]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("center", help="lat,lon（wgs）")
    f.add_argument("--zoom", type=int, default=18, help="17≈1.1m/px 看片区，19≈0.28m/px 看单栋楼")
    f.add_argument("--radius", type=int, default=4, help="中心切片向外扩几圈，4 → 9x9 切片")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--source", choices=list(SOURCES) + ["auto"], default="auto",
                   help="卫星源。auto（默认）先试高德→天地图→Esri→Google，都不通才报错；"
                        "国内直连用 amap/tianditu/esri；google 最锐利但要代理（配了 --proxy 时 auto 会先试它）")
    f.add_argument("--proxy", default=os.environ.get("GEO_PROXY"))
    f.add_argument("--cache", type=Path, default=Path(".geo-cache/tiles"))

    mk = sub.add_parser("mark")
    mk.add_argument("image", type=Path)
    mk.add_argument("--points", type=Path, help="JSON: {name: [lat, lon]}")
    mk.add_argument("--out", type=Path, required=True)
    mk.add_argument("--label", action="store_true")
    mk.add_argument("--geojson", type=Path, action="append", help="叠加 GeoJSON（osm.py geom 的输出），可重复，每个文件一种颜色")
    mk.add_argument("--sector", action="append", help="机位视野扇形 lat,lon,朝向,水平视角,半径m，可重复")

    sh = sub.add_parser("sheet", help="候选点逐个居中出卫星缩略图，拼成带编号的对比页")
    shg = sh.add_mutually_exclusive_group(required=True)
    shg.add_argument("--points", type=Path, help="JSON {name: [lat, lon]}")
    shg.add_argument("--grid", help="s,w,n,e：把一片区域按网格铺满，逐格出图（格名 r行c列：r00 最北、c00 最西；中心坐标写进 <out>.cells.json）")
    sh.add_argument("--step", type=float, help="--grid 的格距（米），默认按每格覆盖范围留一成重叠")
    sh.add_argument("--zoom", type=int, default=18)
    sh.add_argument("--size", type=int, default=320, help="每格像素")
    sh.add_argument("--cols", type=int, default=4, help="每页 cols×cols 格")
    sh.add_argument("--out", type=Path, required=True)
    sh.add_argument("--source", choices=list(SOURCES) + ["auto"], default="auto",
                    help="卫星源，同 fetch（默认 auto = 国内源优先，不需要代理）")
    sh.add_argument("--proxy", default=os.environ.get("GEO_PROXY"))
    sh.add_argument("--cache", type=Path, default=Path(".geo-cache/tiles"))

    pl = sub.add_parser("px2ll", help="拼图（或它的裁剪/缩放图）上的像素 → 经纬度")
    pl.add_argument("image", type=Path, help="tiles.py fetch 出的拼图（旁边要有同名 .json）")
    pl.add_argument("--px", action="append", required=True, help="x,y，可重复；量的是裁剪/缩放后的图时配 --crop/--scale")
    pl.add_argument("--crop", default="0,0", help="你量像素的那张图在原拼图里的左上角 x0,y0")
    pl.add_argument("--scale", type=float, default=1.0, help="那张图相对原拼图的缩放倍数（缩小一半写 0.5）")
    pl.add_argument("--out", type=Path, help="写出 {p1: [lat, lon], ...}")

    args = ap.parse_args(_neg_coords(sys.argv[1:]))
    if args.cmd == "px2ll":
        m = Mosaic(args.image)
        x0, y0 = (float(v) for v in args.crop.split(","))
        pts = {}
        for i, s in enumerate(args.px, 1):
            x, y = (float(v) for v in s.split(","))
            lat, lon = m.to_ll(x0 + x / args.scale, y0 + y / args.scale)
            pts[f"p{i}"] = [round(lat, 6), round(lon, 6)]
            print(f"p{i}  ({x:.0f},{y:.0f}) → {lat:.6f},{lon:.6f}")
        if args.out:
            args.out.write_text(json.dumps(pts, indent=1), encoding="utf-8")
            print(f"-> {args.out}")
        return
    if args.cmd == "sheet":
        pts = json.loads(args.points.read_text(encoding="utf-8")) if args.points else grid_points(args.grid, args.zoom, args.size, args.step)
        if args.grid:
            args.out.with_suffix(".cells.json").write_text(json.dumps(pts, indent=1), encoding="utf-8")
            print(f"网格 {len(pts)} 格（格名 r行c列，行从北往南、列从西往东；每格中心坐标 -> {args.out.with_suffix('.cells.json')}）")
        for p in sheet(pts, args.zoom, args.size, args.cols, args.out, args.source, args.proxy, args.cache):
            print(p)
        return
    if args.cmd == "fetch":
        lat, lon = map(float, args.center.split(","))
        meta = fetch((lat, lon), args.zoom, args.radius, args.out, args.source, args.proxy, args.cache)
        print(json.dumps(meta))
    elif args.cmd == "mark":
        if not (args.points or args.geojson or args.sector):
            ap.error("mark 至少要 --points、--geojson、--sector 之一")
        mark(args.image, json.loads(args.points.read_text(encoding="utf-8")) if args.points else {}, args.out, args.label,
             args.geojson, args.sector)
        print(args.out)


if __name__ == "__main__":
    # 中文 Windows 默认按 GBK 输出：遇到 m²、ñ 会崩，agent 读到的中文也是乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
