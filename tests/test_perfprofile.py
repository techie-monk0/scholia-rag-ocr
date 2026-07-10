"""Unit tests for lib/perfprofile.py — the reusable two-layer profiler.

These deliberately avoid psutil / GPU / real sampling: they exercise the
APPLICATION layer (interpretation heuristics, stage-window slicing, and per-stage
process attribution) against a fake machine layer, plus the pure helpers. So they
pass on any host and lock the contract other phases (OCR, embedding, ...) depend
on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import perfprofile as pp


# -- pure helpers ----------------------------------------------------------- #
def test_stats_basic():
    s = pp._stats([4, 1, 3, 2])
    assert (s["n"], s["min"], s["max"]) == (4, 1, 4)
    assert abs(s["mean"] - 2.5) < 1e-9
    assert s["last"] == 2                    # preserves input order for 'last'


def test_parse_cpu_time_formats():
    assert pp._parse_cpu_time("01:00") == 60
    assert pp._parse_cpu_time("1-00:00:00") == 86400
    assert abs(pp._parse_cpu_time("00:01.50") - 1.5) < 1e-9


def test_make_matcher_variants():
    rec = {"name": "llama-server", "cmd": "/path/llama-server --port 8094"}
    other = {"name": "python", "cmd": "ocr_one.py"}
    assert pp.make_matcher(None)(rec) is True            # all
    assert pp.make_matcher("llama")(rec) is True         # substring
    assert pp.make_matcher("llama")(other) is False
    assert pp.make_matcher(["surya", "llama"])(rec) is True   # any of list
    assert pp.make_matcher(lambda r: r["name"] == "python")(other) is True


# -- interpretation heuristics (deterministic; ncpu/provider fixed in ctx) -- #
def _ctx(provider="off", deltas=None, ncpu=18):
    return {"ncpu": ncpu, "gpu_provider": provider, "deltas": deltas or {}}


def _data(*, proc_cpu=None, sys_cpu=None, gpu_util=None, mem_avail=None):
    machine = {}
    if sys_cpu is not None:
        machine["sys.cpu_pct"] = {"mean": sys_cpu}
    if gpu_util is not None:
        machine["gpu.util_pct"] = {"mean": gpu_util}
    if mem_avail is not None:
        machine["sys.mem_avail_gb"] = {"min": mem_avail}
    stage_procs = {}
    if proc_cpu is not None:
        stage_procs["cpu_pct"] = {"mean": proc_cpu}
    return {"machine": machine, "stage_procs": stage_procs}


def test_interpret_cpu_bound_uses_stage_procs():
    # CPU-bound is judged on the STAGE's own processes, not the whole machine.
    d = _data(proc_cpu=18 * 100 * 0.9, gpu_util=5.0)
    assert pp.default_interpret(d, _ctx("ioreg"))["bottleneck"] == "cpu"


def test_interpret_gpu_bound_nvml():
    d = _data(gpu_util=85.0, proc_cpu=200.0)
    assert pp.default_interpret(d, _ctx("nvml"))["bottleneck"] == "gpu"


def test_interpret_ioreg_understates_but_still_gpu():
    d = _data(gpu_util=55.0, proc_cpu=100.0)
    v = pp.default_interpret(d, _ctx("ioreg"))
    assert v["bottleneck"] == "gpu"
    assert any("machine-wide" in n for n in v["notes"])


def test_interpret_holding_vram_is_not_gpu_bound():
    # low utilisation must not read as gpu-bound even with the model resident.
    d = _data(gpu_util=8.0, proc_cpu=50.0)
    assert pp.default_interpret(d, _ctx("ioreg"))["bottleneck"] == "underutilised"


def test_interpret_swapping_is_memory():
    d = _data(proc_cpu=50.0)
    v = pp.default_interpret(d, _ctx(deltas={"sys.swap_used_gb": 0.5}))
    assert v["bottleneck"] == "memory" and v["swapping"] is True


def test_interpret_low_free_ram_is_memory():
    d = _data(proc_cpu=50.0, mem_avail=1.0)
    assert pp.default_interpret(d, _ctx())["bottleneck"] == "memory"


# -- application layer: stage windows + per-stage process attribution -------- #
class _FakeSampler:
    """Stand-in machine layer: a fixed series + the attrs report() reads."""
    root_pid = 123
    interval = 1.0
    gpu_provider = "off"

    def __init__(self, samples):
        self._samples = samples

    def snapshot(self):
        return list(self._samples)


def _sample(t, procs):
    return {"t": t, "sys.cpu_pct": 5.0, "procs": procs}


def test_stage_window_and_process_attribution():
    # In-window sample: a busy llama-server + an idle helper. The stage matches
    # only "llama", so stage_procs must reflect the server, not the helper, and
    # the tree must reflect both.
    busy = {1: {"name": "llama-server", "cmd": "llama-server", "cpu_pct": 600.0,
                "rss_gb": 6.0, "threads": 8.0},
            2: {"name": "python", "cmd": "helper", "cpu_pct": 5.0,
                "rss_gb": 0.5, "threads": 2.0}}
    samples = [_sample(0.0, busy),           # before window
               _sample(1.0, busy),           # inside window
               _sample(2.0, busy)]           # after window
    prof = pp.StageProfiler(sampler=_FakeSampler(samples))
    prof._stages.append(("ocr", 0.5, 1.5, "llama"))   # window + match
    rep = prof.report()
    ocr = rep["stages"]["ocr"]

    assert ocr["samples"] == 1
    # stage-specific procs = only the llama-server
    assert ocr["stage_procs"]["names"] == ["llama-server"]
    assert ocr["stage_procs"]["cpu_pct"]["mean"] == 600.0
    assert ocr["stage_procs"]["nproc"]["max"] == 1
    # whole tree = both processes
    assert ocr["tree"]["cpu_pct"]["mean"] == 605.0
    assert ocr["tree"]["nproc"]["max"] == 2
    # the other two samples fall outside the window
    assert rep["stages"]["(unstaged)"]["samples"] == 2


def test_unmatched_stage_procs_is_empty_but_machine_present():
    procs = {1: {"name": "python", "cmd": "x", "cpu_pct": 10.0, "rss_gb": 0.1,
                 "threads": 1.0}}
    prof = pp.StageProfiler(sampler=_FakeSampler([_sample(1.0, procs)]))
    prof._stages.append(("embed", 0.5, 1.5, "no-such-proc"))
    embed = prof.report()["stages"]["embed"]
    assert embed["stage_procs"]["names"] == []
    assert embed["stage_procs"]["cpu_pct"]["mean"] == 0.0
    assert "sys.cpu_pct" in embed["machine"]   # overall machine still reported


def test_shared_sampler_not_owned_is_not_stopped():
    prof = pp.StageProfiler(sampler=_FakeSampler([]))
    assert prof._owns_sampler is False
    prof.start(); prof.stop()                  # no-op, must not crash


# -- detector layer (modular, between machine and application) --------------- #
def _gpu_samples(points, stage=None):
    """points: list of (t, gpu) -> tagged-sample dicts (stage optional)."""
    out = []
    for t, gpu in points:
        s = {"t": float(t), "t_wall": 1000.0 + t, "gpu.util_pct": float(gpu)}
        if stage is not None:
            s["stage"] = stage
        out.append(s)
    return out


def test_dip_detector_classifies_cold_start_vs_stall():
    # idle at the START of an ocr block = cold-start; idle later = stall.
    cold = _gpu_samples([(0, 0), (5, 0), (10, 0), (15, 90)], stage="ocr")
    f = pp.DipDetector(min_seconds=8).scan(cold)
    assert len(f) == 1 and f[0]["kind"] == "cold_start" and f[0]["severity"] == "info"

    busy = _gpu_samples([(t, 90) for t in range(0, 200, 5)], stage="ocr")
    late = _gpu_samples([(200, 0), (205, 0), (210, 0)], stage="ocr")
    f = pp.DipDetector(min_seconds=8).scan(busy + late)
    assert len(f) == 1 and f[0]["kind"] == "stall" and f[0]["severity"] == "warn"


def test_dip_detector_cpu_stage_is_expected():
    s = _gpu_samples([(0, 0), (5, 0), (10, 0)], stage="assembly")
    f = pp.DipDetector(min_seconds=8).scan(s)
    assert len(f) == 1 and f[0]["kind"] == "expected_cpu_stage"


def test_dip_detector_ignores_short_blip():
    s = _gpu_samples([(0, 0)], stage="ocr")          # single sample, dur 0
    assert pp.DipDetector(min_seconds=10).scan(s) == []


def test_dip_detector_is_metric_agnostic():
    # point it at a non-GPU metric to prove the framework is generic.
    s = [{"t": float(t), "t_wall": 1000.0 + t, "stage": "ocr",
          "proc.cpu_pct": v} for t, v in [(0, 0), (5, 0), (10, 0), (15, 800)]]
    f = pp.DipDetector(metric="proc.cpu_pct", below=50, min_seconds=8).scan(s)
    assert len(f) == 1 and f[0]["metric"] == "proc.cpu_pct"


def test_stageprofiler_runs_registered_detectors():
    samples = _gpu_samples([(0, 0), (5, 0), (10, 0), (15, 90), (20, 92)])
    prof = pp.StageProfiler(sampler=_FakeSampler(samples))
    prof.add_detector(pp.DipDetector(min_seconds=8))
    prof._stages.append(("ocr", -1.0, 30.0, None))   # window tags all as ocr
    findings = prof.report()["findings"]
    assert any(f["detector"] == "dip" and f["kind"] == "cold_start"
               for f in findings)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
