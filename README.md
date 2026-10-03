# Place finding project

**看图找地点** —— 给一张照片，输出经得起核对的拍摄地：坐标 + 误差半径、相机朝向、证据图、分档置信度。

无文字、无车牌、无地标也能定位。原理是拿 OpenStreetMap 几何、高程天际线、卫星影像
和街景做证据匹配，而不是猜。

```
photo → 27.335 → 171 → 14,372 → 22 → 3 → 1 → ±2 m
（像素 → 方位角 → 距离 → 交点 → 候选机位 → 约束条数 → 独立验证 → 误差半径）
```

## 特点

- **无文字也能定位** — OSM 几何、桥梁间距、影子方位、山脊线都能当证据
- **结论可核对** — 每个结论都指出跑了哪条命令、产出了哪个文件
- **脚本排序，模型裁定** — 脚本先算先排，模型只在自动排好的前几名里判断
- **国内开箱可用** — 国内数据源直连，不需要代理（详见下方「国内环境」）

## 安装

需要一个支持 Agent Skill 的客户端（Claude Code / Codex / Cursor / Gemini CLI /
OpenCode / GitHub Copilot 等），以及 Python 3.10+ 和 [`uv`](https://docs.astral.sh/uv/)。

```bash
npx skills add Oldcircle/geo-sleuth
```

手动安装：

```bash
git clone https://github.com/leonfighting-py/geo-sleuth-cn
mkdir -p ~/.agents/skills
cp -r geo-sleuth-cn/skills/geo-sleuth ~/.agents/skills/
```

`~/.agents/skills/` 可被 Codex、Cursor、Gemini CLI、OpenCode、GitHub Copilot 共用。

## 用法

把照片丢给 agent，说一句「这张图是在哪拍的」就行。agent 会读 `SKILL.md`、
跑脚本、回来给出拍摄点、朝向和卫星证据图。

也可以直接当命令行工具用：

```bash
export CLAUDE_SKILL_DIR=~/.agents/skills/geo-sleuth

# 第 0 步：探网络（10 秒，强烈建议先跑）
uv run $CLAUDE_SKILL_DIR/scripts/netcheck.py

# 第 1 步：元数据 + OCR + 以图搜图 + 边缘图，一步到位
uv run $CLAUDE_SKILL_DIR/scripts/intake.py photo.jpg --out-dir intake/
```

## 国内环境

这是本仓库相对上游的主要改动：**让 skill 在国内无代理环境下也能用**。

```bash
uv run <skill>/scripts/netcheck.py
```

会打印出每个数据源当前通不通，以及代理该怎么配。典型输出：

```
已自动发现本机代理：http://127.0.0.1:7892
  ✓ 高德卫星瓦片     HTTP 200
  ✓ 百度识图         HTTP 200
  ✓ Google 卫星瓦片   直连不通，走代理可以（14451 B）
```

**直连可用**：高德卫星/路网瓦片、百度全景、百度识图、360 POI、AWS 高程切片、
必应国内版、搜狗图片、HF 模型镜像

**需要代理**：Google 卫星、Google 街景、OSM Overpass、Yandex 识图

也就是说**国内任务不配代理也能定位到城市 / 景区 / 片区级**；
配了代理可以进一步到路 / 楼级。`netcheck.py` 会自动发现本机代理并实测筛选，
不用手填端口。

详细说明见 [`改造说明.md`](./改造说明.md)，里面还记录了几个容易踩的坑
（高德瓦片的坐标系口径、huggingface endpoint patch 等）。

## 工具箱

20 个脚本，每个只做一件事：

| 脚本 | 作用 |
|---|---|
| `netcheck.py` | **探测各数据源连通性 + 代理配置指引（先跑这个）** |
| `hf_sources.py` | 模型下载源自动回落（国内镜像） |
| `intake.py` | 元数据 + 边缘图 + 变体 + OCR + 以图搜图，一条命令 |
| `tiles.py` | 卫星切片拼图、候选点缩略图、标点、视野扇形 |
| `sat_scan.py` | CLIP 零样本排序卫星图候选点 |
| `match.py` | DINOv2 + SIFT 排名照片与实景图相似度 |
| `terrain.py` | 高程天际线渲染、山脊提取、机位拟合 |
| `geo.py` | 坐标换算、方位距离、视线交会、机位反解 |
| `osm.py` | OSM Overpass 要素共现、线变点、走廊 |
| `baidu_pano.py` | 百度全景（国内街景） |
| `poi.py` | 地名 → 坐标候选 |
| `board.py` | 候选盘：证据似然比、排名、出结论检查 |
| `sun.py` | 太阳位置、影长比、受光面定朝向 |
| `imgprep.py` | 放大读字、四边四角、搜图变体、等间距构件提取 |
| `clues.py` | 查表：车牌、区号、国家码、行驶方向 |
| `evidence.py` | 证据图生成 |
| `pose.py` | 多点反解机位（经纬度、高度、朝向、俯仰） |
| `ocr.py` | 读照片文字（Apple Vision / RapidOCR） |
| `gsv.py` | Google 街景（国外，需代理） |
| `gazetteer.py` | 行政区名录 |

## 置信度分级

| 级别 | 判据 |
|---|---|
| 城市 | 文字、车牌、区号或确认过的锚点 |
| 片区 | 唯一地标在卫星图上对上 ≥3 项俯视特征 |
| 路 | 实景对上 ≥2 项独有特征 |
| 楼 | 两条独立约束 + 实景 ≥3 项 |
| 楼层 | 两种参照交叉 |

**定不到点就给区间 + 缺什么信息**，不硬猜、不取中点、不画大圆。

## 数据来源

- OpenStreetMap 贡献者（ODbL）— 脚本实时查询，不随仓库分发
- AWS Terrain Tiles（Terrarium 高程）
- 百度地图（全景、识图、POI）
- 高德地图（卫星与路网瓦片）
- DINOv2（Meta AI）、CLIP（OpenAI）
- 行政区划数据来自 [modood/Administrative-divisions-of-China](https://github.com/modood/Administrative-divisions-of-China)
  （`data/` 下的表格由 Wikipedia 整理，CC BY-SA 4.0）

## 许可

MIT（见 [LICENSE](./geo-sleuth/LICENSE)）。
`data/` 中由 Wikipedia 整理的表格采用 CC BY-SA 4.0。

**负责任使用**：只分析你自己拍摄或获得授权的照片，不要用来定位未经同意的人。

## 上游

本仓库基于 [Oldcircle/geo-sleuth](https://github.com/Oldcircle/geo-sleuth) v2
的国内化改造。方法论来自上游，联网数据源、代理处理和模型下载做了针对国内网络的适配。
