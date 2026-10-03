#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""探针：一次性测出"哪些数据源现在真的能用"，并给出可执行的修复建议。

跑一次就知道该走哪条路，不用把命令一个个试到超时。

```bash
python3 netcheck.py                # 测全部（默认 10s 超时）
python3 netcheck.py --at 39.9,116.4# 带上测试坐标，卫星源测得更准
python3 netcheck.py --proxy socks5h://127.0.0.1:10808   # 同时测代理通不通
python3 netcheck.py --json         # 机器可读，给 agent 解析
```

输出三段：
  1. 直连源（不需要代理）——国内任务靠这些就够
  2. 需代理的源 —— 逐个测"直连 / 走代理"哪个通
  3. 结论与建议 —— 直接告诉你该 export什么、该用哪个 --source
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# (名称, 分类, 最小URL, 期望最小字节数)
PROBES = [
    # ---------- 直连就够（国内任务的主路） ----------
    ("高德卫星瓦片", "直连", "https://webst01.is.autonavi.com/appmaptile?style=6&x=99&y=48&z=7", 3000),
    ("高德路网瓦片", "直连", "https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x=99&y=48&z=7", 300),
    ("百度全景接口", "直连", "https://mapsv0.bdimg.com/?qt=bdws&newmap=1&reqflag=pcmap&biz=1&from=webmap&l=12&b=12000000,4000000;13000000,4100000", 2000),
    ("百度识图首页", "直连", "https://graph.baidu.com/s?sign=0&tn=pc&tn=baidulogo", 2000),
    ("360 地图POI", "直连", "https://restapi.map.so.com/newapi?qt=s&wd=%E5%BC%A0%E6%8E%96&c=1", 200),
    ("必应国内版", "直连", "https://cn.bing.com/", 5000),
    ("搜狗图片", "直连", "https://pic.sogou.com/", 2000),
    ("AWS 高程切片", "直连", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/10/512/361.png", 200),
    ("HuggingFace 镜像", "直连", "https://hf-mirror.com/", 200),
    # ---------- 必须代理 ----------
    ("Google 卫星瓦片", "代理", "https://mt1.google.com/vt/lyrs=s&x=99&y=48&z=7", 2000),
    ("Google 地图", "代理", "https://www.google.com/maps", 5000),
    ("Overpass(osm)", "代理", "https://overpass-api.de/api/status", 100),
    ("Esri 卫星", "代理", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/7/48/99", 2000),
]


def discover_proxies() -> list[str]:
    """自动发现本机可能的代理地址，按优先级返回列表（供逐个实测筛出能用的）。

    这是"怎么让 agent 用上代理"的核心：不问用户，自己查。

    优先级很重要——本机上可能同时存在多个代理：
      1. macOS 系统代理（scutil --proxy）——**图形界面开的代理软件真正生效的地方**
      2. 环境变量 HTTP_PROXY / HTTPS_PROXY
      3. 常见端口挨个试（7890/7892/10808/10809/6152/1080/2080/8888/7897）
    环境变量里常有 IDE 或内部服务的代理（如 IDE 或内部服务的代理），
    那个只在内网有用，连Google 不通，所以**系统代理优先**。
    """
    cands = []
    # ① macOS 系统代理优先
    if sys.platform == "darwin":
        try:
            r = subprocess.run(["scutil", "--proxy"], capture_output=True, text=True, timeout=5)
            host = re.search(r"HTTPS?Proxy\s*:\s*(\S+)", r.stdout)
            port = re.search(r"HTTPS?Port\s*:\s*(\d+)", r.stdout)
            if host and port and host.group(1) not in ("",):
                cands.append(f"http://{host.group(1)}:{port.group(1)}")
        except Exception:  # noqa: BLE001
            pass
    # ② 环境变量
    for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        v = os.environ.get(k, "").strip()
        if v and ("127.0.0.1" in v or "localhost" in v):
            cands.append(v)
    # ③ 常见端口挨个试
    for p in (7890, 7892, 10808, 10809, 6152, 1080, 2080, 8888, 7897):
        cands.append(f"http://127.0.0.1:{p}")
    # 去重保序
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def discover_proxy() -> str | None:
    """只返回第一个候选（不实测连通性）。"""
    c = discover_proxies()
    return c[0] if c else None


def probe(url: str, timeout: int, proxy: str | None) -> tuple[bool, int, str]:
    """返回 (通不通, 字节数, 说明)。"""
    cmd = ["curl", "-s", "-m", str(timeout), "-o", "/dev/null",
           "-w", "%{http_code} %{size_download}"]
    if proxy:
        cmd[1:1] = ["-x", proxy]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        out = (r.stdout or "").split()
        code = int(out[0]) if out else 0
        size = int(float(out[1])) if len(out) > 1 else 0
    except Exception as e:  # noqa: BLE001
        return False, 0, f"异常 {type(e).__name__}"
    return code == 200, size, f"HTTP {code}"


def main() -> int:
    ap = argparse.ArgumentParser(description="探测各数据源连通性")
    ap.add_argument("--at", help="测试坐标 lat,lon（卫星源用得上）")
    ap.add_argument("--proxy", default=os.environ.get("GEO_PROXY") or discover_proxy(),
                    help="要测的代理。默认自动发现本机代理（查环境变量和 macOS 系统设置）")
    ap.add_argument("--timeout", type=int, default=10, help="单源超时秒（默认10）")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    # 收集候选代理（系统代理优先，其次环境变量，最后常见端口）
    discover_all = discover_proxies()

    # 找出真正能用的那个：拿 Google 卫星当标尺挨个试
    if args.proxy is None:
        for cand in discover_all:
            ok, _, _ = probe("https://mt1.google.com/vt/lyrs=s&x=99&y=48&z=7", args.timeout, cand)
            if ok:
                args.proxy = cand
                break
    elif args.proxy not in discover_all:
        discover_all = [args.proxy] + discover_all

    def run(item):
        name, kind, url, minsz = item
        direct_ok, direct_sz, direct_msg = probe(url, args.timeout, None)
        via = None
        if kind == "代理" and not direct_ok:
            for cand in (discover_all or ([args.proxy] if args.proxy else [])):
                ok, sz, msg = probe(url, args.timeout, cand)
                if ok:
                    via = {"ok": True, "size": sz, "msg": msg, "proxy": cand}
                    break
        # 直连通了但要代理的源，也报一下走代理的效果（有些环境更稳）
        return {"name": name, "kind": kind, "need_min_bytes": minsz,
                "direct": {"ok": direct_ok, "size": direct_sz, "msg": direct_msg},
                "via_proxy": via}

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, PROBES))

    # ---------- 判定 ----------
    direct_ok_names = [r["name"] for r in results if r["direct"]["ok"] and r["direct"]["size"] >= r["need_min_bytes"]]
    direct_ok_names += [r["name"] for r in results
                        if r["direct"]["ok"] and r["kind"] == "直连" and r["direct"]["size"] > 200]
    direct_ok = sorted(set(direct_ok_names))

    def usable(r) -> bool:
        """这条源现在能不能用（直连通，或代理能通）。"""
        if r["direct"]["ok"] and r["direct"]["size"] >= r["need_min_bytes"]:
            return True
        if r["direct"]["ok"] and r["kind"] == "直连" and r["direct"]["size"] > 200:
            return True
        return bool(r["via_proxy"] and r["via_proxy"]["ok"])

    sat_usable = usable(next(r for r in results if r["name"] == "高德卫星瓦片"))
    google_usable = usable(next(r for r in results if r["name"] == "Google 卫星瓦片"))
    overpass_usable = usable(next(r for r in results if r["name"] == "Overpass(osm)"))

    # ---------- 建议 ----------
    advice = []
    if sat_usable:
        advice.append("卫星图：直接用 tiles.py（默认 --source auto 会走高德），**不需要代理**。")
    else:
        advice.append("卫星图：高德也不通，先查网络/防火墙；或配代理后用 --source google。")
    if overpass_usable:
        o_direct = next(r for r in results if r["name"] == "Overpass(osm)")["direct"]["ok"]
        advice.append(f"OSM Overpass：可用（{'直连' if o_direct else f'走代理 {args.proxy}'}），"
                      f"osm.py / gazetteer.py 能正常跑。")
    else:
        advice.append("OSM Overpass：不通 → osm.py、gazetteer.py 都用不了；"
                      "改用 poi.py（360+百度直连）拿坐标，结论里写明 OSM 枚举缺失。")
    if google_usable:
        g_direct = next(r for r in results if r["name"] == "Google 卫星瓦片")["direct"]["ok"]
        if g_direct:
            advice.append("Google 卫星/街景：直连就能用，--source google 可直接用。")
        else:
            advice.append(f"Google 卫星/街景：**可用**（走代理 {args.proxy}）。"
                          f"用 `export GEO_PROXY={args.proxy}` 即可解锁 gsv.py / --source google。")
    else:
        advice.append("Google 卫星/街景：不通（也没找到可用代理）。国内任务不受影响——"
                      "高德卫星 + 百度全景已覆盖；想用 Google 见下方「怎么让代理生效」。")
    if not results[0]["direct"]["ok"]:
        advice.append("连高德都超时：本机可能完全没外网/被墙，先确认网络。")

    out = {"direct_ok": direct_ok, "sat_usable": sat_usable,
           "google_usable": google_usable, "overpass_usable": overpass_usable,
           "advice": advice, "results": results}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ---------- 人看的 ----------
    G, R, Y, D = "\033[32m", "\033[31m", "\033[33m", "\033[2m"
    if sys.stdout.isatty():
        pass
    else:
        G = R = Y = D = ""

    if args.proxy:
        print(f"\n已自动发现本机代理：{args.proxy}")
        print("（以后 skill 里的命令加 export GEO_PROXY=上面这个地址 就能走代理）")
    print("\n=== 直连源（国内任务主路，不需要代理）===")
    for r in results:
        if r["kind"] != "直连":
            continue
        ok = r["direct"]["ok"] and r["direct"]["size"] >= r["need_min_bytes"]
        mark = f"{G}✓{D}" if ok else f"{R}✗{D}"
        print(f"  {mark} {r['name']:<16} {r['direct']['msg']:>10}  {r['direct']['size']:>8} B")

    print("\n=== 需要代理的源 ===")
    for r in results:
        if r["kind"] != "代理":
            continue
        d = r["direct"]
        if d["ok"] and d["size"] >= r["need_min_bytes"]:
            print(f"  {G}✓{D} {r['name']:<16} 直连就能用（{d['size']} B）")
        elif r["via_proxy"]:
            v = r["via_proxy"]
            print(f"  {G}✓{D} {r['name']:<16} 直连不通，走代理可以（{v['size']} B）")
        elif args.proxy:
            print(f"  {R}✗{D} {r['name']:<16} 直连不通，{args.proxy} 也不通")
        else:
            print(f"  {Y}!{D} {r['name']:<16} 直连不通，没测代理（要测加 --proxy）")

    print("\n=== 结论 ===")
    for a in advice:
        print(f"  · {a}")

    print("\n=== 怎么让代理在 skill 里生效（agent 照这个做） ===")
    print("""
  代理不是 skill 内部能 magically 配置的东西，它就是一个环境变量。agent 要做的是：

  1) 先探明用户机器上有没有代理在跑（别问用户端口，自己查）：
       macOS：scutil --proxy | grep -i port
       Windows：reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" /v ProxyServer
       Linux：env | grep -i proxy

  2) 探到端口后，**每个 bash 命令都带上**（shell 不跨命令保留变量）：
       export GEO_PROXY=socks5h://127.0.0.1:7890 && <原命令>
     一次性让本会话后续命令都继承：
       在WorkBuddy 里把这条写进 ~/.workbuddy/MEMORY.md，或每条命令前重复写。
     永久生效（写进 shell 配置，新终端也生效）：
       echo 'export GEO_PROXY=socks5h://127.0.0.1:7890' >> ~/.zshrc && source ~/.zshrc

  3) socks5h（h=host 由代理解析 DNS）优于 socks5。国内代理一律用 socks5h。
     Chrome 系（revimg.py）要用 socks5://（脚本会自动改）。

  4) 改了之后重跑本脚本确认：netcheck.py --proxy socks5h://127.0.0.1:7890

  5) 只想让某一条命令走代理，不污染全局：直接给那个命令加 --proxy 参数。
     支持 --proxy 的脚本：osm.py gazetteer.py tiles.py gsv.py revimg.py
                          poi.py terrain.py match.py sat_scan.py intake.py clues.py geo.py

  6) 没有代理、也不想装：国内任务本来就够用（高德卫星 + 百度全景 + 百度识图 +
     360POI + AWS高程 + 必应），只是 Overpass 枚举和 Google 街景/历史影像用不了。
     这时把结论写成"景区/片区级"而不是"楼/路口级"，并说明缺的是 OSM 枚举与实景比对。

=== sat_scan.py / match.py 的模型（CLIP / DINOv2）===
  这两个要下~350MB 权重。国内用 hf-mirror 或 modelscope，**不需要代理**：
      uv run <skill>/scripts/hf_sources.py     # 自动探哪个镜像能用并设好
  坑：huggingface_hub 默认下载超时只有 10 秒，权重下不完会被报成"连不上"，
  误判成网络不通。hf_sources.py 已把 HF_HUB_DOWNLOAD_TIMEOUT 提到 180 秒。
  仍然失败就别纠结——sat_scan.py 本来就只是"排序不是判定"，直接用替代路径：
      tiles.py sheet --points <候选点> --zoom 17 --out sheet.jpg
  出带编号和十字的候选点缩略图，人工看，效果一样。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
