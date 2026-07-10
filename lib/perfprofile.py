"""perfprofile.py — a small, reusable, two-layer performance profiler.

The design is split so the same sampling substrate serves any workload (OCR,
embedding, indexing, ...):

  MACHINE LAYER  ``MachineSampler``
      Samples raw stats on a background thread and exposes a timestamped series.
      Each sample carries THREE views: whole-machine (CPU/mem/swap), the GPU
      (machine-wide), and a PER-PROCESS breakdown of the monitored process tree
      (pid -> name/cmd/cpu%/rss/threads). It knows nothing about stages or what
      the numbers mean — it just measures, and gives enough detail for a caller
      to attribute usage however it likes.

  APPLICATION LAYER  ``StageProfiler``
      Records which STAGE was active over which wall-clock window (and, per
      stage, which processes are "its own" via a name/cmd ``match``), slices the
      machine series by stage, and reports per stage: the OVERALL machine view,
      the whole monitored TREE, and the STAGE-SPECIFIC processes — then
      INTERPRETS (cpu/gpu/memory-bound, swapping). Swap ``interpreter`` to give a
      phase its own reading.

WHY NOT AN OFF-THE-SHELF TOOL: CPU/memory sampling leans on ``psutil`` when
present (the standard; not worth reinventing) but degrades to a stdlib path so
the module imports with zero install. The custom parts are (a) GPU sampling on
Apple Silicon WITHOUT sudo, for which no library exists (asitop/macmon need
``sudo powermetrics``) — we read ``ioreg`` directly — (b) the layer split, and
(c) per-stage process attribution. Each GPU number records its PROVIDER.

USAGE — instrument phases, attribute each to its own processes::

    from perfprofile import StageProfiler
    with StageProfiler(interval=1.0) as prof:            # this proc + children
        with prof.stage("ocr", match=["llama-server", "surya"]):
            ocr(...)
        with prof.stage("embedding", match="embed_worker"):
            embed(...)
    print(prof.summary())                                # overall + per-stage procs
    prof.write_json("perf.json")

USAGE — just sample, interpret elsewhere (machine layer only)::

    s = MachineSampler(root_pid=PID, interval=1.0).start()
    ... ; series = s.snapshot(); s.stop()                # each: {'t', sys.*, gpu.*, 'procs'}

USAGE — monitor an already-running tree from the shell::

    python3 perfprofile.py --pid 28298 --interval 1 --duration 30 [--json out]
    python3 perfprofile.py --pid 28298 --raw             # machine layer, no verdict

MACHINE METRICS (a metric that can't be read is absent -> NaN -> dropped):
  sys.cpu_pct      whole-machine CPU %, 0..100 (psutil; else load1-derived)
  sys.mem_used_gb / sys.mem_avail_gb / sys.swap_used_gb
  gpu.util_pct     GPU utilisation % (machine-wide; SEE provider caveats)
  gpu.mem_gb / gpu.power_w
PER-PROCESS (sample["procs"][pid]): name, cmd, cpu_pct (0..cores*100 summed),
  rss_gb, threads (psutil only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

try:                                    # the fast, accurate path when installed
    import psutil
except Exception:                       # pragma: no cover - optional dependency
    psutil = None

_GB = 1024 ** 3
_IS_DARWIN = platform.system() == "Darwin"


# =========================================================================== #
# Collectors: each reads one family of machine stats. read() returns a dict; an
# unavailable value is simply absent. Collectors must never raise out of read().
# =========================================================================== #
class SystemCollector:
    """Whole-machine CPU%, physical memory, and swap (flat metrics)."""

    def __init__(self):
        if psutil is not None:
            psutil.cpu_percent(None)        # prime: first call is a throwaway

    def read(self) -> dict:
        out = {}
        if psutil is not None:
            try:
                out["sys.cpu_pct"] = psutil.cpu_percent(None)
                vm = psutil.virtual_memory()
                out["sys.mem_used_gb"] = (vm.total - vm.available) / _GB
                out["sys.mem_avail_gb"] = vm.available / _GB
                out["sys.swap_used_gb"] = psutil.swap_memory().used / _GB
            except Exception:
                pass
            return out
        try:
            ncpu = os.cpu_count() or 1
            out["sys.cpu_pct"] = min(100.0, 100.0 * os.getloadavg()[0] / ncpu)
        except OSError:
            pass
        out.update(_darwin_memory() if _IS_DARWIN else _linux_memory())
        return out


class ProcTreeCollector:
    """Per-process breakdown of a root pid's subtree. read() returns
    ``{"procs": {pid: {name, cmd, cpu_pct, rss_gb, threads}}}`` so the caller can
    sum the whole tree OR just the processes it cares about. CPU% is normalised
    to 0..(cores*100) per process, summed across the tree by consumers."""

    def __init__(self, root_pid: int):
        self.root_pid = root_pid
        self._procs = {}                    # pid -> psutil.Process (cpu% state)
        self._prev_cpu = {}                 # pid -> (cpu_seconds, wall) for stdlib
        if psutil is not None:
            self._refresh_psutil()

    def read(self) -> dict:
        return {"procs": self._read_psutil() if psutil is not None
                else self._read_stdlib()}

    # -- psutil path -------------------------------------------------------- #
    def _tree_procs(self):
        try:
            root = psutil.Process(self.root_pid)
        except Exception:
            return []
        procs = [root]
        try:
            procs += root.children(recursive=True)
        except Exception:
            pass
        return procs

    def _refresh_psutil(self):
        for p in self._tree_procs():
            if p.pid not in self._procs:
                try:
                    p.cpu_percent(None)     # prime baseline
                except Exception:
                    continue
                self._procs[p.pid] = p

    def _read_psutil(self) -> dict:
        self._refresh_psutil()
        out, live = {}, {}
        for pid, p in self._procs.items():
            try:
                with p.oneshot():
                    cpu = p.cpu_percent(None)
                    rss = p.memory_info().rss
                    threads = float(p.num_threads())
                    name = p.name()
                    try:
                        cmd = " ".join(p.cmdline())[:300] or name
                    except Exception:
                        cmd = name
                out[pid] = {"name": name, "cmd": cmd, "cpu_pct": cpu,
                            "rss_gb": rss / _GB, "threads": threads}
                live[pid] = p
            except Exception:
                continue                    # exited between refresh and read
        self._procs = live
        return out

    # -- stdlib path (parse `ps`) ------------------------------------------- #
    def _read_stdlib(self) -> dict:
        rows = _ps_snapshot()               # pid -> (ppid, cpu_seconds, rss_kb, cmd)
        if not rows:
            return {}
        kids = {}
        for pid, (ppid, _c, _r, _cmd) in rows.items():
            kids.setdefault(ppid, []).append(pid)
        seen, stack = set(), [self.root_pid]
        while stack:                        # DFS the subtree from root_pid
            pid = stack.pop()
            if pid in seen or pid not in rows:
                continue
            seen.add(pid)
            stack.extend(kids.get(pid, []))
        now = time.time()
        out = {}
        for pid in seen:
            _ppid, csec, rss_kb, cmd = rows[pid]
            cpu = 0.0
            prev = self._prev_cpu.get(pid)
            if prev is not None and now > prev[1]:
                cpu = 100.0 * (csec - prev[0]) / (now - prev[1])
            self._prev_cpu[pid] = (csec, now)
            first = cmd.split()[0] if cmd.split() else str(pid)
            out[pid] = {"name": os.path.basename(first), "cmd": cmd,
                        "cpu_pct": max(0.0, cpu), "rss_gb": rss_kb * 1024 / _GB,
                        "threads": float("nan")}
        return out


class GpuCollector:
    """GPU utilisation / memory / power, via the best provider for this host.

    Provider precedence and what each sees:
      * ``nvml``        — NVIDIA via pynvml: util, memory, power. Accurate.
      * ``powermetrics``— macOS, OPT-IN, needs sudo: real GPU active residency
                          (the trustworthy compute-load number on Apple Silicon).
      * ``ioreg``       — macOS, no sudo: GPU memory is reliable, but the
                          utilisation counter is GRAPHICS-PIPELINE based and
                          UNDERSTATES Metal/compute load. Use it for memory + a
                          rough floor on util.
    All providers report the WHOLE device (machine-wide), not per process.
    ``self.provider`` records which produced the numbers."""

    def __init__(self, prefer: str = "auto"):
        self.provider = "none"
        self._nvml = None
        if prefer in ("auto", "nvml"):
            self._try_nvml()
        if self.provider == "none" and _IS_DARWIN and prefer == "powermetrics" \
                and shutil.which("powermetrics"):
            self.provider = "powermetrics"
        if self.provider == "none" and _IS_DARWIN and prefer in ("auto", "ioreg") \
                and shutil.which("ioreg"):
            self.provider = "ioreg"

    def _try_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.provider = "nvml"
        except Exception:
            self._nvml = None

    def read(self) -> dict:
        try:
            if self.provider == "nvml":
                return self._read_nvml()
            if self.provider == "ioreg":
                return _ioreg_gpu()
            if self.provider == "powermetrics":
                return _powermetrics_gpu()
        except Exception:
            pass
        return {}

    def _read_nvml(self) -> dict:
        n = self._nvml
        u = n.nvmlDeviceGetUtilizationRates(self._nvml_h)
        m = n.nvmlDeviceGetMemoryInfo(self._nvml_h)
        out = {"gpu.util_pct": float(u.gpu), "gpu.mem_gb": m.used / _GB}
        try:
            out["gpu.power_w"] = n.nvmlDeviceGetPowerUsage(self._nvml_h) / 1000.0
        except Exception:
            pass
        return out


# --------------------------------------------------------------------------- #
# Platform helpers (stdlib parsing) — kept small and defensive.
# --------------------------------------------------------------------------- #
def _ps_snapshot() -> dict:
    """pid -> (ppid, cpu_seconds, rss_kb, command) for every process."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss=,time=,command="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            rows[int(parts[0])] = (int(parts[1]), _parse_cpu_time(parts[3]),
                                   float(parts[2]), parts[4])
        except ValueError:
            continue
    return rows


def _parse_cpu_time(s: str) -> float:
    """ps TIME field ([[dd-]hh:]mm:ss[.cc]) -> seconds."""
    days = 0.0
    if "-" in s:
        d, s = s.split("-", 1)
        days = float(d)
    try:
        nums = [float(b) for b in s.split(":")]
    except ValueError:
        return 0.0
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return days * 86400 + sec


def _darwin_memory() -> dict:
    out = {}
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True).stdout)
    except Exception:
        total = 0
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))

        def pages(label):
            m = re.search(rf"{label}:\s+(\d+)", vm)
            return int(m.group(1)) if m else 0
        free = (pages("Pages free") + pages("Pages inactive")
                + pages("Pages speculative")) * page
        if total:
            out["sys.mem_used_gb"] = (total - free) / _GB
            out["sys.mem_avail_gb"] = free / _GB
    except Exception:
        pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                            capture_output=True, text=True).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", sw)
        if m:
            out["sys.swap_used_gb"] = float(m.group(1)) / 1024.0
    except Exception:
        pass
    return out


def _linux_memory() -> dict:
    out = {}
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = float(v.strip().split()[0]) * 1024   # kB -> bytes
        total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
        out["sys.mem_used_gb"] = (total - avail) / _GB
        out["sys.mem_avail_gb"] = avail / _GB
        out["sys.swap_used_gb"] = (info.get("SwapTotal", 0)
                                   - info.get("SwapFree", 0)) / _GB
    except Exception:
        pass
    return out


def _ioreg_gpu() -> dict:
    out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                         capture_output=True, text=True, timeout=5).stdout
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    mem = re.search(r'"In use system memory"=(\d+)', out)
    res = {}
    if util:
        res["gpu.util_pct"] = float(util.group(1))
    if mem:
        res["gpu.mem_gb"] = float(mem.group(1)) / _GB
    return res


def _powermetrics_gpu() -> dict:
    """One powermetrics GPU sample (needs sudo). Active residency is the honest
    Apple-Silicon GPU-load number; ``util_pct`` here means active-residency %."""
    out = subprocess.run(
        ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-n", "1",
         "-i", "200"], capture_output=True, text=True, timeout=10).stdout
    res = {}
    m = re.search(r"GPU (?:HW )?active residency:\s+([\d.]+)%", out)
    if m:
        res["gpu.util_pct"] = float(m.group(1))
    m = re.search(r"GPU Power:\s+([\d.]+)\s*mW", out)
    if m:
        res["gpu.power_w"] = float(m.group(1)) / 1000.0
    return res


# =========================================================================== #
# MACHINE LAYER — sample raw stats on a thread; no stage / interpretation logic.
# =========================================================================== #
class MachineSampler:
    """Background sampler of machine + GPU + per-process-tree stats.

    ``root_pid`` defaults to this process. ``interval`` is seconds/sample.
    ``gpu`` selects the provider ('auto'|'ioreg'|'powermetrics'|'nvml'|'off').
    Each sample: ``{'t': perf_counter, sys.*, gpu.*, 'procs': {pid: {...}}}``."""

    def __init__(self, root_pid: int | None = None, interval: float = 1.0,
                 gpu: str = "auto", on_sample=None):
        self.root_pid = root_pid if root_pid is not None else os.getpid()
        self.interval = max(0.05, float(interval))
        self._collectors = [SystemCollector(), ProcTreeCollector(self.root_pid)]
        self.gpu = GpuCollector(prefer=gpu) if gpu != "off" else None
        if self.gpu is not None:
            self._collectors.append(self.gpu)
        self.on_sample = on_sample          # optional: called with each sample
        self._samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def gpu_provider(self) -> str:
        return getattr(self.gpu, "provider", "off")

    def read_now(self) -> dict:
        """One synchronous sample. The background loop is the normal path —
        calling this ad hoc perturbs the per-process CPU% deltas.

        Each sample carries BOTH clocks: ``t`` (perf_counter, monotonic) for
        stage-window math, and ``t_wall`` (epoch seconds) so a slice can be lined
        up against a wall-clock task log or merged across processes/runs."""
        s = {"t": time.perf_counter(), "t_wall": time.time()}
        for c in self._collectors:
            try:
                s.update(c.read())
            except Exception:
                pass
        return s

    def start(self) -> "MachineSampler":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop,
                                            name="perfprofile", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> "MachineSampler":
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval * 2 + 2)
            self._thread = None
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def _loop(self):
        while not self._stop.is_set():
            s = self.read_now()
            with self._lock:
                self._samples.append(s)
            if self.on_sample is not None:
                try:
                    self.on_sample(s)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._samples)

    def window(self, t0: float, t1: float) -> list:
        with self._lock:
            return [s for s in self._samples if t0 <= s["t"] <= t1]

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._samples[-1]) if self._samples else None


# =========================================================================== #
# APPLICATION LAYER — stage windows + per-stage process attribution + verdicts.
# =========================================================================== #
def make_matcher(match):
    """Build a predicate over a process record {name, cmd, ...}. ``match`` may be
    None (all procs), a substring, a list of substrings (case-insensitive, against
    name+cmd), or a callable predicate."""
    if match is None:
        return lambda rec: True
    if callable(match):
        return match
    needles = [match] if isinstance(match, str) else list(match)
    needles = [n.lower() for n in needles]

    def f(rec):
        hay = (rec.get("name", "") + " " + rec.get("cmd", "")).lower()
        return any(n in hay for n in needles)
    return f


def default_interpret(data: dict, ctx: dict) -> dict:
    """Heuristic, provider-aware verdict for one stage.

    Attribution: CPU-bound is judged on the STAGE'S OWN processes (data
    ['stage_procs']); GPU and memory/swap are machine-wide (data['machine']),
    because no per-process GPU accounting exists here. Returns {bottleneck,
    swapping, notes}; bottleneck in cpu|gpu|memory|mixed|underutilised|unknown."""
    machine, sp = data.get("machine", {}), data.get("stage_procs", {})
    ncpu = ctx.get("ncpu") or 1
    prov = ctx.get("gpu_provider", "off")
    deltas = ctx.get("deltas", {})
    notes = []

    # Intermittency (sawtooth) — a high idle fraction split across episodes means
    # the resource cycled idle<->busy rather than ran flat; surface it because a
    # plain mean (e.g. "gpu 50%") would read as steady when it's really stalling.
    gd = (data.get("duty", {}) or {}).get("gpu", {})
    idle_frac = gd.get("idle_frac")
    if idle_frac is not None and idle_frac >= 0.2:
        notes.append(f"intermittent: GPU idle {idle_frac * 100:.0f}% of stage in "
                     f"{gd['episodes']} episode(s) (longest "
                     f"{gd['max_episode_samples']} samples) — periodic stall "
                     "(e.g. checkpoint cold-start), not a steady ceiling")

    proc_cpu = (sp.get("cpu_pct") or {}).get("mean")
    sys_cpu = (machine.get("sys.cpu_pct") or {}).get("mean")
    gpu_util = (machine.get("gpu.util_pct") or {}).get("mean")
    mem_avail = (machine.get("sys.mem_avail_gb") or {}).get("min")
    cpu_ceiling = ncpu * 100.0

    swapping = deltas.get("sys.swap_used_gb", 0.0) > 0.05
    mem_bound = swapping or (mem_avail is not None and mem_avail < 2.0)
    if swapping:
        notes.append(f"swap grew {deltas['sys.swap_used_gb']:+.2f} GB during "
                     "stage — active swapping")
    if mem_avail is not None and mem_avail < 2.0:
        notes.append(f"only {mem_avail:.1f} GB RAM free at tightest")

    # CPU-bound is attributed to the stage's own processes, not the whole machine.
    cpu_bound = ((proc_cpu is not None and proc_cpu >= 0.85 * cpu_ceiling)
                 or (sys_cpu is not None and sys_cpu >= 85.0))

    # GPU metrics are machine-wide; only THIS stage's bottleneck when the stage's
    # processes own the GPU work (true for OCR: llama-servers are in the tree).
    gpu_bound = False
    if gpu_util is not None:
        notes.append("gpu.* is machine-wide (not attributed to the stage's procs)")
    if prov in ("nvml", "powermetrics"):
        gpu_bound = gpu_util is not None and gpu_util >= 70.0
    elif prov == "ioreg":
        if gpu_util is not None and gpu_util >= 50.0:   # understated -> strong LB
            gpu_bound = True
            notes.append("GPU util via ioreg understates Metal/compute — true "
                         "utilisation is likely higher than shown")

    if mem_bound:
        bottleneck = "memory"
    elif gpu_bound and cpu_bound:
        bottleneck = "mixed"
    elif gpu_bound:
        bottleneck = "gpu"
    elif cpu_bound:
        bottleneck = "cpu"
    elif proc_cpu is not None or gpu_util is not None:
        bottleneck = "underutilised"
    else:
        bottleneck = "unknown"
    return {"bottleneck": bottleneck, "swapping": swapping,
            "idle_frac": idle_frac, "notes": notes}


# Machine metrics whose first->last change in a window is meaningful.
_DELTA_METRICS = ("sys.swap_used_gb", "sys.mem_avail_gb", "gpu.mem_gb")


# =========================================================================== #
# DETECTOR LAYER — sits between the machine sampler and the application's report.
# A Detector consumes the tagged sample series and emits structured findings;
# register your own with StageProfiler.add_detector(...). Detectors are
# independent, generic (point them at any sampled metric), and unit-testable in
# isolation. The dip detector below is the first; shard-imbalance, swapping,
# thermal-throttle, etc. slot in the same way.
# =========================================================================== #
_SENTINEL = object()


class Detector:
    """Base class. Override ``scan(samples) -> list[finding]``. ``samples`` are
    the tagged machine samples (each: t, t_wall, stage, sys.*/gpu.* metrics,
    procs). A finding is a JSON-able dict carrying at least detector/kind/stage/
    severity, so it flows into the report and the [PERF] output unchanged."""

    name = "detector"

    def scan(self, samples: list) -> list:
        return []


class DipDetector(Detector):
    """Sustained dips of ``metric`` below ``below``, classified by WHERE they
    fall: an expected CPU-stage idle (rasterize/assembly), an OCR cold-start
    (near the start of an ocr block), or a mid-stream STALL (the actionable one).
    Generic — aim it at gpu.util_pct, a proc cpu series, anything sampled."""

    name = "dip"

    def __init__(self, metric: str = "gpu.util_pct", below: float = 10.0,
                 min_seconds: float = 10.0, coldstart_window_s: float = 120.0,
                 cpu_stages=("rasterize", "assembly")):
        self.metric = metric
        self.below = below
        self.min_seconds = min_seconds
        self.coldstart_window_s = coldstart_window_s
        self.cpu_stages = tuple(cpu_stages)

    def scan(self, samples: list) -> list:
        samples = sorted(samples, key=lambda s: s.get("t", 0))
        findings, run = [], []
        block_start, prev_stage = 0, _SENTINEL
        for s in samples:
            st = s.get("stage")
            if st != prev_stage:                # new (stage) block starts here
                prev_stage, block_start = st, s.get("t", 0)
            s["_block_start"] = block_start
            v = s.get(self.metric)
            if v is not None and v < self.below:
                run.append(s)
            else:
                self._emit(run, findings)
                run = []
        self._emit(run, findings)
        return findings

    def _emit(self, run, findings):
        if not run:
            return
        dur = run[-1].get("t", 0) - run[0].get("t", 0)
        if dur < self.min_seconds:              # ignore short blips
            return
        stage = run[0].get("stage")
        since = run[0].get("t", 0) - run[0].get("_block_start", run[0].get("t", 0))
        if stage in self.cpu_stages:
            kind, sev = "expected_cpu_stage", "info"
        elif stage == "ocr" and since <= self.coldstart_window_s:
            kind, sev = "cold_start", "info"
        else:
            kind, sev = "stall", "warn"
        vals = [s.get(self.metric) for s in run if s.get(self.metric) is not None]
        findings.append({
            "detector": self.name, "kind": kind, "metric": self.metric,
            "stage": stage, "severity": sev,
            "iso": time.strftime("%H:%M:%S",
                                 time.localtime(run[0].get("t_wall", 0))),
            "t_wall": round(run[0].get("t_wall", 0), 2),
            "duration_s": round(dur, 1),
            "mean": round(sum(vals) / len(vals), 1) if vals else None})


class StageProfiler:
    """Drive a MachineSampler; per stage, report the OVERALL machine, the whole
    monitored TREE, and the STAGE-SPECIFIC processes, then interpret.

    ``stage(label, match=...)`` names which processes are the stage's own (a
    substring / list / predicate over name+cmd; default = the whole tree). Pass
    an existing ``sampler`` to share one machine layer; swap ``interpreter`` to
    give a phase its own reading."""

    def __init__(self, sampler: MachineSampler | None = None, *,
                 interpreter=None, on_sample=None, detectors=None, **sampler_kw):
        self.sampler = sampler or MachineSampler(**sampler_kw)
        self._owns_sampler = sampler is None
        self.interpret = interpreter or default_interpret
        self.detectors = list(detectors) if detectors else []
        self._stages = []                   # (label, t_enter, t_exit, match)
        self._current = None                # live stage label (for streaming)
        self._lock = threading.Lock()
        # Live streaming: tag each machine sample with the current stage and hand
        # it to the user sink as (sample, stage_label) — e.g. to emit one
        # [PERF] line per sample into a task log, no post-hoc join needed.
        self._user_sink = on_sample
        if on_sample is not None:
            self.sampler.on_sample = self._emit

    def _emit(self, sample):
        if self._user_sink is not None:
            try:
                self._user_sink(sample, self._current or "(idle)")
            except Exception:
                pass

    def enter_stage(self, label):
        """Set the live stage label streamed with subsequent samples. (Window
        boundaries for the post-hoc report are still set by stage()/record().)"""
        self._current = label

    def add_detector(self, detector) -> "StageProfiler":
        """Register a Detector (the modular middle layer). It runs over the
        tagged sample series at report() time and contributes findings."""
        self.detectors.append(detector)
        return self

    def _run_detectors(self) -> list:
        if not self.detectors:
            return []
        samples = self.tagged_samples()
        out = []
        for d in self.detectors:
            try:
                out.extend(d.scan(samples))
            except Exception:
                pass
        return out

    def start(self) -> "StageProfiler":
        if self._owns_sampler:
            self.sampler.start()
        return self

    def stop(self) -> "StageProfiler":
        if self._owns_sampler:
            self.sampler.stop()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    @contextmanager
    def stage(self, label: str, match=None):
        """Mark the window where ``label`` is active; ``match`` selects the
        stage's own processes for attribution (see make_matcher)."""
        t_enter = time.perf_counter()
        self._current = label
        try:
            yield
        finally:
            self.record(label, t_enter, time.perf_counter(), match)
            self._current = None

    def record(self, label: str, t_enter: float, t_exit: float, match=None):
        """Record a stage window directly (non-context-manager form), for a
        pipeline that already has its own stage-boundary perf_counter marks and
        can't easily wrap each block in ``with``."""
        with self._lock:
            self._stages.append((label, t_enter, t_exit, match))

    def tagged_samples(self) -> list:
        """The raw machine series with a 'stage' label added to each sample (the
        stage whose window contains its timestamp, else '(idle)'). Lets the
        machine data be persisted at full per-sample resolution for offline
        analysis (stalls, interleaving) — not just the aggregated report."""
        samples = self.sampler.snapshot()
        with self._lock:
            stages = list(self._stages)

        def stage_at(t):
            for label, a, b, _m in stages:
                if a <= t <= b:
                    return label
            return "(idle)"
        out = []
        for s in samples:
            row = dict(s)
            row["stage"] = stage_at(s["t"])
            out.append(row)
        return out

    # -- reporting ---------------------------------------------------------- #
    def _profile(self, wins, sel, matcher, ncpu, prov) -> dict:
        sel = sorted(sel, key=lambda s: s["t"])
        # (1) overall machine: the flat sys.*/gpu.* metrics.
        machine = {}
        for s in sel:
            for k, v in s.items():
                if "." not in k or v is None or _isnan(v):   # metrics are sys.*/gpu.*
                    continue
                machine.setdefault(k, []).append(float(v))
        machine_stats = {k: _stats(v) for k, v in sorted(machine.items())}
        # (2) whole tree + (3) stage-specific processes, per sample then aggregated.
        tree_cpu, tree_rss, tree_n = [], [], []
        sp_cpu, sp_rss, sp_n, sp_names = [], [], [], {}
        for s in sel:
            procs = s.get("procs") or {}
            tree_cpu.append(sum(r["cpu_pct"] for r in procs.values()))
            tree_rss.append(sum(r["rss_gb"] for r in procs.values()))
            tree_n.append(float(len(procs)))
            matched = [r for r in procs.values() if matcher(r)]
            sp_cpu.append(sum(r["cpu_pct"] for r in matched))
            sp_rss.append(sum(r["rss_gb"] for r in matched))
            sp_n.append(float(len(matched)))
            for r in matched:
                sp_names[r["name"]] = sp_names.get(r["name"], 0) + 1
        deltas = {k: round(machine[k][-1] - machine[k][0], 3)
                  for k in _DELTA_METRICS if machine.get(k)}
        if tree_rss:
            deltas["tree.rss_gb"] = round(tree_rss[-1] - tree_rss[0], 3)
        # Duty-cycle: expose idle troughs an average would hide (the sawtooth).
        duty = {}
        if machine.get("gpu.util_pct"):
            duty["gpu"] = duty_cycle(machine["gpu.util_pct"], 10.0)
        if sp_cpu:
            duty["stage_procs_cpu"] = duty_cycle(sp_cpu, 15.0)
        data = {
            "duration_s": round(sum(b - a for a, b in wins), 2),
            "samples": len(sel),
            "machine": machine_stats,
            "tree": _proc_block(tree_cpu, tree_rss, tree_n),
            "stage_procs": {**_proc_block(sp_cpu, sp_rss, sp_n),
                            "names": sorted(sp_names)},
            "duty": duty,
            "deltas": deltas,
        }
        ctx = {"ncpu": ncpu, "gpu_provider": prov, "deltas": deltas}
        data["interpretation"] = self.interpret(data, ctx)
        return data

    def report(self) -> dict:
        samples = self.sampler.snapshot()
        ncpu = os.cpu_count() or 1
        prov = self.sampler.gpu_provider
        with self._lock:
            stages = list(self._stages)
        groups = {}                         # label -> {"wins":[...], "match":...}
        for label, a, b, match in stages:
            g = groups.setdefault(label, {"wins": [], "match": match})
            g["wins"].append((a, b))

        def in_any(t):
            return any(a <= t <= b for g in groups.values() for a, b in g["wins"])
        profiles = {}
        for label, g in groups.items():
            sel = [s for s in samples
                   if any(a <= s["t"] <= b for a, b in g["wins"])]
            profiles[label] = self._profile(g["wins"], sel,
                                            make_matcher(g["match"]), ncpu, prov)
        unstaged = [s for s in samples if not in_any(s["t"])]
        if unstaged:
            profiles["(unstaged)"] = self._profile([], unstaged,
                                                   make_matcher(None), ncpu, prov)
        return {
            "root_pid": self.sampler.root_pid,
            "interval_s": self.sampler.interval,
            "gpu_provider": prov,
            "psutil": psutil is not None,
            "ncpu": ncpu,
            "total_samples": len(samples),
            "stages": profiles,
            "findings": self._run_detectors(),
        }

    def write_json(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.report(), fh, indent=2)

    def summary(self) -> str:
        """Compact human table: per stage a verdict, the overall machine view,
        and the stage's own processes."""
        rep = self.report()
        out = [f"perf profile  (root pid {rep['root_pid']}, "
               f"{rep['interval_s']}s interval, {rep['total_samples']} samples, "
               f"gpu={rep['gpu_provider']}, "
               f"psutil={'yes' if rep['psutil'] else 'no'}, "
               f"{rep['ncpu']} cores)"]

        def g(stats, metric, key="mean"):
            return (stats.get(metric) or {}).get(key)

        for stage, prof in rep["stages"].items():
            v = prof["interpretation"]
            m, sp = prof["machine"], prof["stage_procs"]
            out.append("")
            out.append(f"  [{stage}]  {prof['duration_s']}s, {prof['samples']} "
                       f"samples  ->  BOTTLENECK: {v['bottleneck'].upper()}"
                       + ("  +SWAPPING" if v["swapping"] else ""))
            for note in v["notes"]:
                out.append(f"      · {note}")
            out.append("      machine:  cpu {:>5}% (max {:>3})   gpu {:>5}% "
                       "(max {:>3})   mem_free {:>5} GB   swap {:>5} GB".format(
                           _fmt(g(m, "sys.cpu_pct")), _fmt(g(m, "sys.cpu_pct", "max")),
                           _fmt(g(m, "gpu.util_pct")), _fmt(g(m, "gpu.util_pct", "max")),
                           _fmt(g(m, "sys.mem_avail_gb", "min")),
                           _fmt(g(m, "sys.swap_used_gb", "max"))))
            names = ", ".join(sp.get("names", [])) or "(none matched)"
            out.append("      procs:    cpu {:>6}% (max {:>6})   rss {:>6} GB "
                       "(max {:>6})   n={}  [{}]".format(
                           _fmt(g(sp, "cpu_pct")), _fmt(g(sp, "cpu_pct", "max")),
                           _fmt(g(sp, "rss_gb")), _fmt(g(sp, "rss_gb", "max")),
                           int(g(sp, "nproc", "max") or 0), names))
        findings = rep.get("findings", [])
        if findings:
            out.append("")
            out.append(f"  detectors — {len(findings)} finding(s):")
            for f in findings:
                out.append(f"    [{f.get('detector')}/{f.get('kind')}] "
                           f"{f.get('stage')} @{f.get('iso')} ~{f.get('duration_s')}s "
                           f"{f.get('metric')} mean {f.get('mean')} "
                           f"({f.get('severity')})")
        return "\n".join(out)


# Convenience alias for the common case (instrument-and-interpret).
PerfProfiler = StageProfiler


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _stats(values) -> dict:
    vs = sorted(values)
    n = len(vs)
    if not n:
        return {"n": 0, "min": float("nan"), "mean": float("nan"),
                "max": float("nan"), "p95": float("nan"), "last": float("nan")}
    mean = sum(vs) / n
    p95 = vs[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    return {"n": n, "min": round(vs[0], 3), "mean": round(mean, 3),
            "max": round(vs[-1], 3), "p95": round(p95, 3),
            "last": round(values[-1], 3)}


def _proc_block(cpu, rss, nproc) -> dict:
    """Aggregate per-sample tree/stage sums into cpu/rss/nproc stat blocks."""
    return {"cpu_pct": _stats(cpu), "rss_gb": _stats(rss), "nproc": _stats(nproc)}


def duty_cycle(series, idle_below: float) -> dict:
    """Duty-cycle of a per-sample series: what fraction of the time it sat idle
    (below ``idle_below``) and in how many separate episodes. This is what turns
    a misleading "mean 50%" into "idle 48% across 13 episodes" — i.e. it exposes
    a sawtooth (e.g. per-checkpoint cold-start stalls) that an average hides."""
    n = len(series)
    if not n:
        return {}
    idle = [v < idle_below for v in series]
    runs, cur = [], 0
    for x in idle:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    busy = [v for v, i in zip(series, idle) if not i]
    return {"idle_frac": round(sum(idle) / n, 3), "episodes": len(runs),
            "mean_episode_samples": round(sum(runs) / len(runs), 1) if runs else 0,
            "max_episode_samples": max(runs) if runs else 0,
            # mean over the non-idle samples — the saturation level DURING work,
            # separate from one-time warmup/idle that drags the plain mean down.
            "busy_mean": round(sum(busy) / len(busy), 1) if busy else None}


def _fmt(v) -> str:
    return "-" if v is None or _isnan(v) else f"{float(v):.1f}"


# --------------------------------------------------------------------------- #
# Decision layer: read per-book [PERF] SUMMARY records from a task log and turn
# them into an actionable digest (throughput + GPU-during-work + a corrective
# recommendation), and compare runs (e.g. 2 vs 3 instances).
# --------------------------------------------------------------------------- #
def _agg_ocr(summaries: list) -> dict:
    """Aggregate the ocr-stage summaries of one run: page-weighted s/page, mean
    GPU-during-work, mean idle fraction, tightest free RAM, any swapping."""
    pages = sum(r.get("pages") or 0 for r in summaries)
    sp_w = sum((r.get("s_per_page") or 0) * (r.get("pages") or 0) for r in summaries)
    busy = [r["gpu_busy_mean"] for r in summaries if r.get("gpu_busy_mean") is not None]
    idle = [r["gpu_idle_frac"] for r in summaries if r.get("gpu_idle_frac") is not None]
    mem = [r["mem_avail_min_gb"] for r in summaries
           if r.get("mem_avail_min_gb") is not None]
    return {"books": len(summaries), "pages": pages,
            "s_per_page": round(sp_w / pages, 2) if pages else None,
            "gpu_busy": round(sum(busy) / len(busy), 1) if busy else None,
            "idle_frac": round(sum(idle) / len(idle), 3) if idle else 0.0,
            "mem_min": min(mem) if mem else None,
            "swapping": any(r.get("swapping") for r in summaries)}


def recommend_action(agg: dict) -> str:
    """Heuristic corrective action from one run's ocr aggregate. Order matters:
    rule out memory pressure, then stalls, then read the GPU-during-work level."""
    if agg["swapping"] or (agg["mem_min"] is not None and agg["mem_min"] < 2):
        return "REDUCE instances — memory pressure / swapping"
    if agg["idle_frac"] and agg["idle_frac"] >= 0.2:
        return (f"FIX STALLS first — GPU idle {agg['idle_frac'] * 100:.0f}% of OCR "
                "(cold-start / detection<->recognition bubbles); not an "
                "instance-count problem")
    b = agg["gpu_busy"]
    if b is not None and b >= 88:
        return ("GPU SATURATED during work — do NOT add instances (contention). "
                "Faster needs lower per-page cost (--max-decode / model) or better "
                "detection<->recognition interleaving")
    if b is not None and b <= 70:
        return ("GPU HAS HEADROOM during work — try +1 instance or larger "
                "--parallel (watch memory bandwidth)")
    return "BALANCED — hold instance count"


def _analyze(logs: list) -> int:
    """Print a per-book + aggregate OCR decision digest for each log, then (for
    >1 log) compare runs by wall throughput."""
    runs = {}
    for log in logs:
        recs = []
        try:
            for line in open(log, errors="ignore"):
                if "[PERF] " not in line:
                    continue
                try:
                    r = json.loads(line.split("[PERF] ", 1)[1])
                except ValueError:
                    continue
                if r.get("summary") and r.get("stage") == "ocr":
                    recs.append(r)
        except OSError as e:
            print(f"cannot read {log}: {e}")
            continue
        runs[log] = recs
        print(f"\n=== {log} : {len(recs)} finished book(s) ===")
        if not recs:
            print("  (no completed-book OCR summaries yet — books still running)")
            continue
        print(f"  {'book':<30}{'pages':>6}{'s/pg':>7}{'gpu_busy':>9}"
              f"{'idle%':>7}  bottleneck")
        for r in recs:
            print(f"  {str(r.get('book'))[:30]:<30}{r.get('pages', '-'):>6}"
                  f"{_fmt(r.get('s_per_page')):>7}{_fmt(r.get('gpu_busy_mean')):>9}"
                  f"{_fmt((r.get('gpu_idle_frac') or 0) * 100):>7}  "
                  f"{r.get('bottleneck', '?')}")
        a = _agg_ocr(recs)
        print(f"  AGG  s/pg {_fmt(a['s_per_page'])}  gpu_busy {_fmt(a['gpu_busy'])}%"
              f"  idle {_fmt(a['idle_frac'] * 100)}%  mem_min {_fmt(a['mem_min'])}GB")
        print(f"  -> {recommend_action(a)}")
    ready = [(log, _agg_ocr(r)) for log, r in runs.items() if r]
    if len(ready) > 1:
        ready.sort(key=lambda x: (x[1]["s_per_page"] or float("inf")))
        print("\n=== comparison (fastest wall throughput wins) ===")
        for log, a in ready:
            print(f"  {log}: s/pg {_fmt(a['s_per_page'])}, "
                  f"gpu_busy {_fmt(a['gpu_busy'])}%, idle {_fmt(a['idle_frac']*100)}%")
        print(f"  -> use the config of {ready[0][0]} "
              f"(s/pg {_fmt(ready[0][1]['s_per_page'])})")
    return 0


# --------------------------------------------------------------------------- #
# CLI: monitor an external process tree (no code instrumentation needed).
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Sample CPU / memory / GPU for a process tree. Default: a "
                    "per-stage profile with interpretation (one 'monitor' "
                    "stage). --raw dumps the machine layer with no verdict.")
    ap.add_argument("--pid", type=int, default=None,
                    help="Root pid of the tree to watch (default: this process).")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds/sample.")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="How long to sample (s) before reporting.")
    ap.add_argument("--match", default=None,
                    help="Substring(s) (comma-separated) selecting the 'monitor' "
                         "stage's own processes (default: the whole tree).")
    ap.add_argument("--gpu", default="auto",
                    choices=["auto", "ioreg", "powermetrics", "nvml", "off"],
                    help="GPU provider (powermetrics needs sudo; ioreg no sudo).")
    ap.add_argument("--raw", action="store_true",
                    help="Machine layer only: print per-sample tree totals.")
    ap.add_argument("--analyze", nargs="+", metavar="LOG", default=None,
                    help="Read per-book [PERF] SUMMARY lines from one or more task "
                         "logs and print an OCR decision digest + corrective "
                         "recommendation (and compare runs, e.g. 2 vs 3 instances).")
    ap.add_argument("--json", type=str, default=None, help="Also write JSON here.")
    args = ap.parse_args(argv)

    if args.analyze:                        # decision layer over existing logs
        return _analyze(args.analyze)

    def sample_for(secs):
        deadline = time.perf_counter() + secs
        while time.perf_counter() < deadline:
            time.sleep(min(0.5, args.interval))

    if args.raw:                            # machine layer, no interpretation
        sampler = MachineSampler(root_pid=args.pid, interval=args.interval,
                                 gpu=args.gpu)
        with sampler:
            try:
                sample_for(args.duration)
            except KeyboardInterrupt:
                pass
        for s in sampler.snapshot():
            procs = s.get("procs") or {}
            row = {k: round(v, 2) for k, v in s.items()
                   if "." in k and not _isnan(v)}            # sys.*/gpu.* metrics
            row = {"time": time.strftime("%H:%M:%S",
                                         time.localtime(s.get("t_wall", 0))), **row}
            row["proc.cpu_pct"] = round(sum(r["cpu_pct"] for r in procs.values()), 1)
            row["proc.rss_gb"] = round(sum(r["rss_gb"] for r in procs.values()), 2)
            row["nproc"] = len(procs)
            print(row)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(sampler.snapshot(), fh, indent=2)
            print(f"\nJSON -> {args.json}")
        return 0

    match = args.match.split(",") if args.match else None
    prof = StageProfiler(root_pid=args.pid, interval=args.interval, gpu=args.gpu)
    with prof, prof.stage("monitor", match=match):
        try:
            sample_for(args.duration)
        except KeyboardInterrupt:
            print("\n(interrupted — reporting what we have)", file=sys.stderr)
    print(prof.summary())
    if args.json:
        prof.write_json(args.json)
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
