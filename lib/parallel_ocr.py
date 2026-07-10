"""Run Surya OCR (and detection) over a folder using N parallel llama-server
instances, then merge the per-shard results.

Why: one llama-server leaves the GPU ~40% idle (the vision pipeline serializes);
2-3 independent server instances fill it for ~1.4-1.9x throughput. Past ~3 the
GPU saturates and it regresses, so callers should keep N small (default 3).

Surya normally spawns and shares ONE server (sentinel/lock at
~/.cache/datalab/surya, not configurable), so to get genuinely separate
instances we launch the servers ourselves and pin each shard's ``surya_ocr`` to
its own server via ``SURYA_INFERENCE_URL`` (the external-attach path skips the
sentinel). Detection (``surya_detect``) needs no server — it just shards.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import surya_backend as sb
import proc_group
from surya.settings import settings
from surya.inference.backends.llamacpp import _download_gguf_files


# --------------------------------------------------------------------------- #
# Servers
# --------------------------------------------------------------------------- #
# Concurrent OCR slots inside each llama-server (the per-instance --parallel).
SLOTS_PER_SERVER = 8


def _ctx_size() -> int:
    return max(16384, SLOTS_PER_SERVER * settings.SURYA_INFERENCE_CTX_PER_SLOT)


def launch_servers(ports, log_dir: Path):
    """Spawn one llama-server per port (vendored arm64 binary). Returns handles."""
    binary = sb._find_llama_server()
    if not binary:
        raise SystemExit("no llama-server binary (vendored .llama/ missing?). "
                         "See ocr_pipeline/README.md.")
    model, mmproj = _download_gguf_files()
    log_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    for port in ports:
        logf = open(log_dir / f"llama_{port}.log", "w")
        cmd = [binary, "-m", model, "--mmproj", mmproj, "-ngl", "99",
               "--host", "127.0.0.1", "--port", str(port),
               "--parallel", str(SLOTS_PER_SERVER), "--ctx-size", str(_ctx_size()),
               "--alias", settings.SURYA_MODEL_CHECKPOINT, "--jinja"]
        # group=False: stay in ocr_one's process group so an ancestor killpg
        # reaps us too; also tracked for the signal/atexit teardown.
        proc = proc_group.popen(cmd, stdout=logf, stderr=logf)
        handles.append((port, proc, logf))
    return handles


def wait_health(port, timeout=300):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"llama-server on port {port} never became healthy")


def kill_servers(handles):
    for _, proc, logf in handles:
        try:
            proc.terminate()
        except Exception:
            pass
        proc_group.reap(proc)
        try:
            logf.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Sharded runs
# --------------------------------------------------------------------------- #
def _resolve_exe(name: str):
    search = os.pathsep.join(sb._bin_dirs() + [os.environ.get("PATH", "")])
    exe = shutil.which(name, path=search)
    if not exe:
        raise SystemExit(f"{name} not found — pip install surya-ocr")
    return exe, search


def _stripe(items, n):
    """Round-robin split (spreads dense/sparse pages evenly across shards)."""
    shards = [[] for _ in range(n)]
    for i, it in enumerate(items):
        shards[i % n].append(it)
    return [s for s in shards if s]


def _fresh_dir(d: Path) -> Path:
    """An empty directory — wiped if it already exists (so a previous run's
    symlinks/outputs can't leak into this one, which would break incremental OCR)."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _shard_dir(work_root: Path, tag: str, k: int, images) -> Path:
    d = _fresh_dir(work_root / f"{tag}_shard{k}")
    for img in images:
        link = d / img.name
        try:
            link.symlink_to(Path(img).resolve())
        except OSError:
            shutil.copy2(img, link)
    return d


def run_sharded(images, n, work_root: Path, *, tool: str, server_urls=None) -> dict:
    """Run ``surya_ocr`` (tool='ocr') or ``surya_detect`` (tool='detect') over
    ``images`` split across ``n`` shards, concurrently. Returns merged
    {image_key: page_data}. For 'ocr', ``server_urls[k]`` pins shard k.
    """
    exe_name = {"ocr": "surya_ocr", "detect": "surya_detect",
                "layout": "surya_layout"}[tool]
    exe, search = _resolve_exe(exe_name)
    shards = _stripe(list(images), n)
    work_root.mkdir(parents=True, exist_ok=True)

    def run_shard(k):
        sd = _shard_dir(work_root, tool, k, shards[k])
        out = _fresh_dir(work_root / f"{tool}_raw{k}")
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PATH"] = search
        if tool in ("ocr", "layout"):              # VLM-backed; surya_detect isn't
            env["SURYA_INFERENCE_URL"] = server_urls[k]
        # Tracked Popen (group=False: stays in ocr_one's group) so a kill of the
        # parent tears this surya_ocr shard down instead of orphaning it.
        proc = proc_group.popen([exe, str(sd), "--output_dir", str(out)], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait()
        finally:
            proc_group.reap(proc)
        return out

    with ThreadPoolExecutor(max_workers=len(shards)) as ex:
        outs = list(ex.map(run_shard, range(len(shards))))

    merged = {}
    for out in outs:
        for rj in out.rglob("results.json"):
            merged.update(json.loads(rj.read_text()))
    return merged
