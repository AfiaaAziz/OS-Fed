"""Network and client-speed emulation (Step 10).

Condition code (section 18 of the plan). Tokens are joined with a dash:
    net0               no limit, no slow clients
    bw<Mbps>           bandwidth of every client's link, e.g. bw10   (1 Mbps = 1,000,000 bit/s)
    d<ms>              delay added to every message, e.g. d50
    slow<n>x<factor>   the LAST n clients are <factor> times slower, e.g. slow2x2
    examples: bw10   d50   bw10-d50   slow2x2   bw10-d50-slow2x2   bw1-d200-slow5x4

Model (write this in the report):
  * every client has its own link to the server; the server's own link is not a bottleneck
  * every message (each download and each upload) costs   bytes*8/(Mbps*1e6) + delay
  * a slow client with factor f pauses (f - 1) x its measured compute time after it finishes training
  * all of this is emulated on ONE computer with time.sleep: it is not a real network test
  * no dropped messages (out of scope)

This module has no torch import, so it is light inside spawned client processes.

Quick calculator (no training needed):
    python -m src.network --condition bw10-d50-slow2x2 --download-bytes 2180392 \
        --upload-bytes 2180392 --compute 3.0 --clients 10
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Columns added to clients.csv by the emulation (append them to the clients.csv column list)
NET_CLIENT_COLUMNS = ["net_download_s", "slow_extra_s", "net_upload_s"]

_BW = re.compile(r"^bw(\d+(?:\.\d+)?)$")
_DELAY = re.compile(r"^d(\d+(?:\.\d+)?)$")
_SLOW = re.compile(r"^slow(\d+)x(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class NetworkConfig:
    bandwidth_mbps: Optional[float] = None   # None = unlimited
    delay_ms: float = 0.0
    n_slow: int = 0
    slow_factor: float = 1.0
    code: str = "net0"

    def describe(self) -> str:
        bw = "unlimited" if self.bandwidth_mbps is None else f"{self.bandwidth_mbps:g} Mbps"
        slow = "none" if self.n_slow == 0 else f"{self.n_slow} client(s) x{self.slow_factor:g} slower"
        return f"{self.code}: bandwidth {bw}, delay {self.delay_ms:g} ms, slow clients {slow}"


def parse_condition(code: str) -> NetworkConfig:
    """'bw10-d50-slow2x2' -> NetworkConfig(...). Raises ValueError on anything unknown."""
    code = (code or "").strip()
    if code in ("", "net0"):
        return NetworkConfig()
    bw = delay = slow = None
    for tok in code.split("-"):
        if tok == "net0":
            raise ValueError("'net0' must be used alone")
        m = _BW.match(tok)
        if m:
            if bw is not None:
                raise ValueError(f"bandwidth given twice in '{code}'")
            bw = float(m.group(1))
            if bw <= 0:
                raise ValueError("bandwidth must be above 0")
            continue
        m = _DELAY.match(tok)
        if m:
            if delay is not None:
                raise ValueError(f"delay given twice in '{code}'")
            delay = float(m.group(1))
            continue
        m = _SLOW.match(tok)
        if m:
            if slow is not None:
                raise ValueError(f"slow clients given twice in '{code}'")
            slow = (int(m.group(1)), float(m.group(2)))
            if slow[1] < 1.0:
                raise ValueError("slowdown factor must be at least 1")
            continue
        raise ValueError(f"unknown token '{tok}' in condition '{code}' (use bw<Mbps>, d<ms>, slow<n>x<factor>)")
    n_slow, factor = slow if slow else (0, 1.0)
    return NetworkConfig(bandwidth_mbps=bw, delay_ms=delay or 0.0, n_slow=n_slow, slow_factor=factor, code=code)


def transfer_seconds(n_bytes: int, bandwidth_mbps: Optional[float], delay_ms: float) -> float:
    """Time for ONE message: bytes*8 / (bandwidth in bit/s) + delay."""
    t = delay_ms / 1000.0
    if bandwidth_mbps is not None:
        t += n_bytes * 8.0 / (bandwidth_mbps * 1e6)
    return t


def slow_extra_seconds(compute_s: float, factor: float) -> float:
    """Extra pause of a slow client: (factor - 1) x its measured compute time."""
    return max(0.0, compute_s * (factor - 1.0))


def slow_client_ids(n_slow: int, n_clients: int) -> Tuple[int, ...]:
    """Rule (decision D23): the slow clients are the n clients with the HIGHEST ids."""
    if n_slow > n_clients:
        raise ValueError(f"{n_slow} slow clients requested but only {n_clients} clients exist")
    return tuple(range(n_clients - n_slow, n_clients))


class NetworkEmulator:
    """Used INSIDE each client process. The pause_* methods sleep and return the measured seconds."""

    def __init__(self, cfg: NetworkConfig, n_clients: int, slow_ids: Optional[Sequence[int]] = None):
        self.cfg = cfg
        self.n_clients = n_clients
        self.slow_ids = tuple(slow_ids) if slow_ids is not None else slow_client_ids(cfg.n_slow, n_clients)

    # ---- pure calculations (also used by Step 11)
    def is_slow(self, client_id: int) -> bool:
        return client_id in self.slow_ids

    def message_seconds(self, n_bytes: int) -> float:
        return transfer_seconds(n_bytes, self.cfg.bandwidth_mbps, self.cfg.delay_ms)

    def extra_seconds(self, client_id: int, compute_s: float) -> float:
        return slow_extra_seconds(compute_s, self.cfg.slow_factor) if self.is_slow(client_id) else 0.0

    def predicted_client_seconds(self, client_id: int, download_bytes: int, upload_bytes: int,
                                 compute_s: float) -> float:
        return (self.message_seconds(download_bytes) + compute_s
                + self.extra_seconds(client_id, compute_s) + self.message_seconds(upload_bytes))

    # ---- real pauses
    @staticmethod
    def _pause(seconds: float) -> float:
        t0 = time.perf_counter()
        if seconds > 0:
            time.sleep(seconds)
        return time.perf_counter() - t0

    def pause_download(self, n_bytes: int) -> float:
        """Call right after the client receives the global model."""
        return self._pause(self.message_seconds(n_bytes))

    def pause_slow(self, client_id: int, compute_s: float) -> float:
        """Call after local training (and snapshot optimization). compute_s = measured compute time."""
        return self._pause(self.extra_seconds(client_id, compute_s))

    def pause_upload(self, n_bytes: int) -> float:
        """Call right before the client puts its message on the queue."""
        return self._pause(self.message_seconds(n_bytes))


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict emulated times without training.")
    ap.add_argument("--condition", default="net0")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--download-bytes", type=int, required=True)
    ap.add_argument("--upload-bytes", type=int, required=True)
    ap.add_argument("--compute", type=float, required=True, help="measured compute seconds of one client per round")
    args = ap.parse_args()
    cfg = parse_condition(args.condition)
    emu = NetworkEmulator(cfg, args.clients)
    print(cfg.describe())
    print(f"slow client ids: {list(emu.slow_ids)}")
    print(f"one download = {emu.message_seconds(args.download_bytes):.3f} s, "
          f"one upload = {emu.message_seconds(args.upload_bytes):.3f} s\n")
    times = [emu.predicted_client_seconds(c, args.download_bytes, args.upload_bytes, args.compute)
             for c in range(args.clients)]
    for c, t in enumerate(times):
        print(f"client {c}: {t:8.3f} s {'(slow)' if emu.is_slow(c) else ''}")
    print(f"\nround time = slowest client = {max(times):.3f} s   (no-limit round = {args.compute:.3f} s)")


if __name__ == "__main__":
    main()
