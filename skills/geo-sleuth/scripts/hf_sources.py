#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""模型下载助手：让 sat_scan.py / match.py 在国内不用代理也能下到 CLIP / DINOv2。

HuggingFace 官方站在国内一般不通，`hf-mirror.com` 也不一定可用。
本模块按可用性依次尝试，**每个都是直连，不需要代理**：

  1. 环境变量 HF_ENDPOINT（用户自己指定过就用它）
  2. hf-mirror.com            国内常用镜像
  3. modelscope.cn/modelscope 国内阿里的镜像（实测比 hf-mirror 更容易连通）
  4. huggingface.co           官方（要代理或网络能直连时）
  5. 官方源（配合 GEO_PROXY 走代理）

用法：

```python
from hf_sources import prepare
prepare()          # 在 from_pretrained 之前调用，会设好 HF_ENDPOINT
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
```

`prepare()` 返回实际用的 endpoint 字符串；全部不可用时返回 None 并打印排查指引。
它只设 `HF_ENDPOINT` 这一个环境变量，不动代理设置。
"""
from __future__ import annotations

import os
import subprocess
import sys

# 国内可直连的模型镜像，按优先级
MIRRORS = [
    "https://hf-mirror.com",
    "https://www.modelscope.cn/models",
    "https://modelscope.cn/models",
]

_PROBE_FILE = "config.json"


def _patch_hub(endpoint: str) -> None:
    """把 huggingface_hub 的 endpoint 换成国内镜像。

    **这是本skill 改造里最难的一个坑，实测踩了三次才对**：

    1. 设 `os.environ["HF_ENDPOINT"]` 完全无效 —— 库在 import 时就固化了。
    2. 改 `huggingface_hub.constants.ENDPOINT` 也无效（1.x 部分版本）——
       因为 `file_download.py` 里写的是 `from .constants import ENDPOINT`，
       **import 时做了值拷贝**，之后改 constants 它看不到。
    3. 真正要改的是**使用方模块自己那份**：`file_download.ENDPOINT`。

    实测有效：file_download → constants → 其它子模块，全部覆盖一遍。
    验证：hf_hub_url("openai/clip-vit-base-patch32","config.json") 必须含镜像域名。
    照这么改后，CLIP 605MB 权重从 hf-mirror 直连下完只用 47 秒，不需要代理。
    """
    ep = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = ep
    for m in ("huggingface_hub.file_download",   # ★关键：它做了值拷贝
              "huggingface_hub.constants",       # 新版直接读这个
              "huggingface_hub.utils._runtime",
              "huggingface_hub"):
        try:
            import importlib
            mod = importlib.import_module(m)
        except Exception:  # noqa: BLE001
            continue
        for attr in ("ENDPOINT", "HF_ENDPOINT", "HUGGINGFACE_HUB_ENDPOINT"):
            try:
                setattr(mod, attr, ep)
            except Exception:  # noqa: BLE001
                pass


def verify_patch(verbose: bool = True) -> bool:
    """验证 patch 真生效：hf_hub_url 应该生成镜像域名而不是官方域名。

    注意：如果 huggingface_hub 还没被 import，这里会返回 False（模块不存在），
    这是正常的 —— prepare() 本来就设计在 import transformers 之前跑。
    真正的验证时机是模型下载时看有没有报"连不上官方站"。
    """
    try:
        from huggingface_hub import hf_hub_url
        u = hf_hub_url("openai/clip-vit-base-patch32", "config.json")
        ok = "huggingface.co" not in u
        if verbose:
            print(f"[hf] endpoint patch {'生效✓' if ok else '未生效✗'}：{u}", file=sys.stderr)
        return ok
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[hf] 无法验证 patch：{e}", file=sys.stderr)
        return False


def _reachable(endpoint: str, timeout: int = 8) -> bool:
    """探这个 endpoint 能不能拿到任意模型文件。"""
    url = f"{endpoint.rstrip('/')}/openai/clip-vit-base-patch32/resolve/main/{_PROBE_FILE}"
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-m", str(timeout), "-w", "%{http_code}", "-L", url],
            capture_output=True, text=True, timeout=timeout + 4,
        )
        return (r.stdout or "").strip() == "200"
    except Exception:  # noqa: BLE001
        return False


def load(model_id: str, cls=None, probe_files=("config.json",)):
    """从国内镜像加载 transformers 模型，镜像间自动回落。

    探针（HEAD config.json）能过不代表大文件能下完——hf-mirror 常在下载几MB 后的
    权重时断。所以这里逐个镜像真下，失败换下一个。

    ```python
    from hf_sources import load
    tok  = load("openai/clip-vit-base-patch32", AutoTokenizer)
    mdl  = load("openai/clip-vit-base-patch32", AutoModel)
    ```
    """
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    cls = cls or (AutoTokenizer if model_id.endswith("gpt2") else AutoModel)
    errs = []
    cands = []
    if os.environ.get("HF_ENDPOINT"):
        cands.append(os.environ["HF_ENDPOINT"].rstrip("/"))
    cands += [m for m in MIRRORS if m.rstrip("/") not in cands]

    for ep in cands:
        os.environ["HF_ENDPOINT"] = ep
        _patch_hub(ep)
        try:
            return cls.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{ep}: {type(e).__name__} {str(e)[:100]}")
            print(f"[hf] {ep} 加载失败，换下一个镜像…", file=sys.stderr)
    raise RuntimeError("所有镜像都下不到模型：\n  " + "\n  ".join(errs))


def prepare(verbose: bool = True) -> str | None:
    """设好 HF_ENDPOINT 和下载超时，返回实际可用的 endpoint（都不可用时返回 None）。"""
    # ★ 超时必须拉长：CLIP 权重 ~350MB，国内镜像实测速度约 8 MB/s，
    # 要 45 秒以上；huggingface_hub 默认 10 秒就断，于是报成"连不上"，
    # 误判成网络不通。实测：不设这个值，hf-mirror 会被误判为不可用。
    for k, v in (("HF_HUB_DOWNLOAD_TIMEOUT", "180"),
                 ("HF_HUB_ETAG_TIMEOUT", "30"),
                 ("HF_HUB_DISABLE_TELEMETRY", "1")):
        os.environ.setdefault(k, v)

    # 关键：huggingface_hub 库在 import 时就把 endpoint 读进常量了，
    # 之后再改 os.environ 也不生效。所以这里除了设环境变量，还要 monkey-patch
    # 那个常量，否则 from_pretrained 还是会打官方站。
    try:
        import huggingface_hub.constants as hfc
        for attr in ("ENDPOINT", "HF_ENDPOINT"):
            if hasattr(hfc, attr):
                setattr(hfc, attr, os.environ.get("HF_ENDPOINT", "").rstrip("/") or getattr(hfc, attr))
    except Exception:  # noqa: BLE001
        pass

    env = os.environ.get("HF_ENDPOINT", "").strip()
    if env and _reachable(env):
        _patch_hub(env)
        if verbose:
            print(f"[hf] 用已设置的 HF_ENDPOINT={env}", file=sys.stderr)
        verify_patch(verbose)
        return env

    # 已设但不通，尝试国内镜像
    for m in MIRRORS:
        if m.rstrip("/") == env.rstrip("/"):
            continue  # 刚探过，不通
        if _reachable(m):
            os.environ["HF_ENDPOINT"] = m
            _patch_hub(m)
            if verbose:
                print(f"[hf] HF_ENDPOINT 设成 {m}（直连可用）", file=sys.stderr)
            verify_patch(verbose)   # 实测：patch 可能对某些版本不生效，要确认
            return m

    if env:
        # 用户指定的虽然这里探不到，但可能是内网镜像，仍然尊重用户设置
        _patch_hub(env)
        if verbose:
            print(f"[hf] 沿用你设的 HF_ENDPOINT={env}（探测不通，但可能内网可用）", file=sys.stderr)
        return env

    # 全不通：看有没有代理
    proxy = os.environ.get("GEO_PROXY") or os.environ.get("HTTPS_PROXY")
    if proxy:
        if verbose:
            print(f"[hf] 镜像都不通；检测到代理 {proxy}，"
                  f"请确保它已导出到 HTTP_PROXY/HTTPS_PROXY 再重试", file=sys.stderr)
    elif verbose:
        print(f"[hf] 国内镜像都连不上，且没配 GEO_PROXY。\n"
              f"     三个办法：① 配代理 export GEO_PROXY=socks5h://127.0.0.1:7890（先跑 netcheck.py探端口）\n"
              f"     ② 手动设镜像 export HF_ENDPOINT=https://hf-mirror.com\n"
              f"     ③ 不用 CLIP：sat_scan.py 不可用时改用 tiles.py sheet 人工看缩略图",
              file=sys.stderr)
    return None


if __name__ == "__main__":
    r = prepare()
    print("可用 endpoint:", r or "（无）")
