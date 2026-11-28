"""Step 10 check. Run from the project root:   python -m scripts.test_network     (takes about 15 seconds)

1. condition codes are parsed correctly and bad codes are rejected
2. THE PLAN'S CHECK: 1,000,000 bytes at 8 Mbps, 0 ms delay  -> about 1.00 s; with 50 ms delay -> about 1.05 s
3. slow-client pause = (factor - 1) x compute time, and only the right clients are slow
4. 4 real client processes pause IN PARALLEL: the round takes the slowest client's time, not the sum
"""
from __future__ import annotations

import multiprocessing as mp
import time

from src.network import (NetworkConfig, NetworkEmulator, parse_condition, slow_client_ids,
                         transfer_seconds)


def approx(value: float, expected: float, tol: float = 0.06) -> bool:
    return abs(value - expected) <= tol


def test_parse() -> None:
    c = parse_condition("bw10-d50-slow2x2")
    assert (c.bandwidth_mbps, c.delay_ms, c.n_slow, c.slow_factor) == (10.0, 50.0, 2, 2.0), c
    assert parse_condition("net0") == NetworkConfig()
    assert parse_condition("d200").bandwidth_mbps is None and parse_condition("d200").delay_ms == 200.0
    assert parse_condition("bw0.5").bandwidth_mbps == 0.5
    assert parse_condition("slow5x4").n_slow == 5
    for bad in ["bw", "bw0", "x10", "bw10-bw5", "net0-bw10", "slow2x0.5", "bw10-d"]:
        try:
            parse_condition(bad)
        except ValueError:
            continue
        raise AssertionError(f"'{bad}' should have been rejected")
    print("PASS condition codes: parsed bw10-d50-slow2x2, net0, d200, bw0.5, slow5x4; 7 bad codes rejected")


def test_formula_and_real_pause() -> None:
    assert transfer_seconds(1_000_000, 8.0, 0.0) == 1.0
    assert abs(transfer_seconds(1_000_000, 8.0, 50.0) - 1.05) < 1e-12
    e0 = NetworkEmulator(parse_condition("bw8"), 10)
    e50 = NetworkEmulator(parse_condition("bw8-d50"), 10)
    t0 = e0.pause_download(1_000_000)
    t50 = e50.pause_upload(1_000_000)
    assert approx(t0, 1.00), t0
    assert approx(t50, 1.05), t50
    none = NetworkEmulator(parse_condition("net0"), 10)
    assert none.pause_download(10**9) < 0.01 and none.pause_upload(10**9) < 0.01
    print(f"PASS 1,000,000 bytes at 8 Mbps: {t0:.3f} s (expected 1.00); with 50 ms delay: {t50:.3f} s (expected 1.05); net0: no pause")


def test_slow_clients() -> None:
    assert slow_client_ids(2, 10) == (8, 9) and slow_client_ids(0, 10) == ()
    emu = NetworkEmulator(parse_condition("slow2x4"), 10)
    assert emu.is_slow(9) and emu.is_slow(8) and not emu.is_slow(0)
    assert emu.extra_seconds(0, 5.0) == 0.0
    assert emu.extra_seconds(9, 5.0) == 15.0            # (4 - 1) x 5 s
    p = emu.pause_slow(9, 0.3)                          # (4 - 1) x 0.3 = 0.9 s
    assert approx(p, 0.9), p
    assert emu.pause_slow(0, 0.3) < 0.01
    print(f"PASS slow clients: ids {list(emu.slow_ids)}; factor 4 on 0.3 s compute paused {p:.3f} s (expected 0.90)")


def _worker(client_id: int, cfg: NetworkConfig, n_clients: int, out) -> None:
    emu = NetworkEmulator(cfg, n_clients)
    t0 = time.perf_counter()
    emu.pause_download(1_000_000)
    t = time.perf_counter(); time.sleep(0.2); compute = time.perf_counter() - t   # stand-in for training
    emu.pause_slow(client_id, compute)
    emu.pause_upload(1_000_000)
    out.put((client_id, time.perf_counter() - t0))


def test_parallel_processes() -> None:
    cfg = parse_condition("bw8-d50-slow2x2")       # message = 1.05 s; slow clients 2 and 3 pause 0.2 s extra
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    t0 = time.perf_counter()
    procs = [ctx.Process(target=_worker, args=(c, cfg, 4, out)) for c in range(4)]
    for p in procs:
        p.start()
    results = dict(out.get() for _ in procs)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    normal, slow = results[0], results[3]
    assert approx(normal, 2.30, 0.12) and approx(slow, 2.50, 0.12), results
    assert wall < 4.5, f"processes did not pause in parallel (wall {wall:.2f}s)"
    print(f"PASS 4 processes in parallel: normal client {normal:.2f} s (expected 2.30), slow client {slow:.2f} s "
          f"(expected 2.50), whole round {wall:.2f} s (not the sum of about 9.4 s)")


if __name__ == "__main__":
    test_parse()
    test_formula_and_real_pause()
    test_slow_clients()
    test_parallel_processes()
    print("\nALL STEP 10 CHECKS PASSED")
