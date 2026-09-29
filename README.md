# Wavelength-recirculating-photonic-memory"""
ABOBB Banking Simulation - a small, checkable result
=====================================================

WHAT THIS IS
------------
A minimal digital-twin simulator for one specific question inside a larger
optical-buffering architecture (Adaptive Banked Optical Block Buffer / ABOBB):
when you split one optical recirculation buffer into N parallel banks, how
does access waiting time actually behave?

This is deliberately small. It does not claim a new physical mechanism --
recirculating optical buffers are established prior art (Burmeister et al.,
Optica 2008; Alexoudi et al., Light: Science and Applications, 2020). The
only claim here is about the QUEUEING BEHAVIOR of splitting one big buffer
into many small ones, checked by simulation rather than assumed.

THE ONE RESULT WORTH YOUR TIME
-------------------------------
Earlier versions of this idea assumed a smooth, monotonically-improving curve
for "waiting time vs. number of banks." This replaces that assumption with an
actual discrete-event M/M/N queueing simulation (Poisson block arrivals, N
banks as parallel servers, shortest-queue scheduling).

Stress-tested at 167% of single-bank capacity (deliberately overloaded, to
make the effect of banking as visible as possible):

    Banks    Simulated mean wait
    1        213,338 ns  -- UNSTABLE, queue grows without bound
    2        149.6 ns
    4        4.7 ns
    8        2.0 ns
    16-64    2.0 ns      -- flat, hits the scheduler-overhead floor

Two findings that matter:
  1. A single overloaded buffer doesn't just get slow -- it's structurally
     UNSTABLE (queue length diverges). Banking changes the buffer's
     stability regime, not just its speed.
  2. Past ~8 banks in this scenario, adding more banks buys NOTHING --
     waiting time flattens at the electronic scheduler-overhead floor
     (2 ns, an input assumption). That floor, not optical physics, is what
     limits further improvement.

ALSO INCLUDED (sanity checks against a prior hand-derived model)
------------------------------------------------------------------
- Loss decay + refresh threshold: matches a prior hand calculation
  (~19.4 circulations to a 0.70 quality threshold at 0.08 dB/circulation).
- WDM capacity scaling: matches the standard C ~= M*C_lambda relationship
  at 100 Gb/s per wavelength.
These aren't the interesting result -- they're here so the banking model's
underlying assumptions (loop delay, capacity) can be checked independently.

EXPLICIT LIMITATIONS (things not yet solved)
-----------------------------------------------
- Exponential service-time and Poisson-arrival assumptions are a standard
  queueing simplification, not a claim about real traffic. Real AI/
  interconnect traffic is bursty, not memoryless -- next step is testing
  under a bursty/self-similar arrival model.
- No optical-quality (BER/OSNR) model yet -- refresh trigger here is
  normalized power only, not a validated signal-quality metric.
- No energy accounting yet -- this says nothing about whether banking is
  worth its control-plane/monitoring overhead in ENERGY, only in latency.
- All parameters (loss, scheduler overhead, service time) are stated
  assumptions below, not measured device numbers.

THE SPECIFIC QUESTION I'D VALUE FEEDBACK ON
-----------------------------------------------
Does the instability-then-floor shape shown here match what you'd expect
from a real optical buffering system, or does a more realistic (bursty)
traffic model change the shape qualitatively rather than just its scale?

HOW TO RUN
----------
    pip install numpy matplotlib
    python3 abobb.py
This regenerates every number above and saves three PNG plots. Nothing
here needs to be taken on trust.

Mahmud Kabir Walid -- independent study project. Not affiliated with,
endorsed by, or reviewed by any company referenced in earlier drafts.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

PARAMS = {
    "group_velocity_m_s": 2e8,
    "line_rate_bps": 100e9,
    "loss_db_per_circulation": 0.08,
    "quality_threshold": 0.70,
    "scheduler_overhead_s": 2e-9,
}


def round_trip_delay(loop_length_m, v_g=PARAMS["group_velocity_m_s"]):
    return 2 * loop_length_m / v_g


def payload_capacity_bits(loop_length_m, line_rate_bps=PARAMS["line_rate_bps"],
                           v_g=PARAMS["group_velocity_m_s"]):
    return line_rate_bps * round_trip_delay(loop_length_m, v_g)


def power_after_n_circulations(n, loss_db=PARAMS["loss_db_per_circulation"]):
    n = np.asarray(n, dtype=float)
    return 10 ** (-n * loss_db / 10)


def circulations_to_threshold(threshold=PARAMS["quality_threshold"],
                               loss_db=PARAMS["loss_db_per_circulation"]):
    return -10 * np.log10(threshold) / loss_db


def simulate_mmn_queue(n_banks, arrival_rate_per_s, service_time_s,
                        scheduler_overhead_s=PARAMS["scheduler_overhead_s"],
                        n_events=20000, seed=0):
    rng = random.Random(seed)
    server_free_at = [0.0] * n_banks
    t = 0.0
    waits = []
    for _ in range(n_events):
        t += rng.expovariate(arrival_rate_per_s)
        idx = min(range(n_banks), key=lambda i: server_free_at[i])
        start = max(t, server_free_at[idx]) + scheduler_overhead_s
        waits.append(start - t)
        server_free_at[idx] = start + rng.expovariate(1 / service_time_s)
    return float(np.mean(waits))


def banking_curve(bank_counts, arrival_rate_per_s, service_time_s):
    return [simulate_mmn_queue(n, arrival_rate_per_s, service_time_s) for n in bank_counts]


def wdm_capacity_bits(loop_length_m, n_wavelengths, line_rate_bps=PARAMS["line_rate_bps"],
                       v_g=PARAMS["group_velocity_m_s"]):
    return n_wavelengths * payload_capacity_bits(loop_length_m, line_rate_bps, v_g)


if __name__ == "__main__":
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    print("=== Stage 1 sanity check ===")
    for L in [1, 10, 100, 1000]:
        t_rt = round_trip_delay(L)
        cap = payload_capacity_bits(L)
        print(f"  {L:>5} m  ->  {t_rt*1e9:8.1f} ns RT delay, {cap/8:10.1f} B capacity")

    n = np.arange(0, 101)
    power = power_after_n_circulations(n)
    thresh = PARAMS["quality_threshold"]
    n_thresh = circulations_to_threshold(thresh)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n, power, lw=2)
    ax.axhline(thresh, ls="--", color="gray", label=f"refresh threshold = {thresh}")
    ax.axvline(n_thresh, ls=":", color="gray")
    ax.set_xlabel("Circulations"); ax.set_ylabel("Normalized optical power")
    ax.set_title(f"Loss decay -> refresh every {n_thresh:.1f} circulations")
    ax.legend(); fig.tight_layout(); fig.savefig("fig1_loss_and_refresh.png")
    print(f"\nRefresh required every {n_thresh:.2f} circulations "
          f"(matches prior hand calculation of ~20).")

    wl_counts = [1, 4, 8, 16, 32]
    caps_KiB = [wdm_capacity_bits(10, m) / 8 / 1024 for m in wl_counts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wl_counts, caps_KiB, "o-", lw=2)
    ax.set_xlabel("WDM wavelengths"); ax.set_ylabel("Circulating payload capacity (KiB)")
    ax.set_title("Idealized WDM capacity scaling, 10 m loop @ 100 Gb/s")
    fig.tight_layout(); fig.savefig("fig2_wdm_capacity.png")

    bank_counts = [1, 2, 4, 8, 16, 32, 64]
    service_time_s = 50e-9
    arrival_rate = 1 / 30e-9
    print(f"\nRunning M/M/N queueing simulation "
          f"(single-bank utilization = {arrival_rate*service_time_s:.2f})...")
    waits_ns = [w * 1e9 for w in banking_curve(bank_counts, arrival_rate, service_time_s)]
    for nb, w in zip(bank_counts, waits_ns):
        print(f"  banks={nb:3d}  mean wait = {w:10.2f} ns")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bank_counts, waits_ns, "o-", lw=2, color="darkred")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.axhline(PARAMS["scheduler_overhead_s"] * 1e9, ls="--", color="gray",
               label="scheduler overhead floor (2 ns)")
    ax.set_xlabel("Number of optical banks")
    ax.set_ylabel("Simulated mean waiting time (ns, log scale)")
    ax.set_title("Banking effect on access waiting -- REAL simulation (M/M/N queue)")
    ax.legend(); fig.tight_layout(); fig.savefig("fig3_banking_simulated.png")

    print("\nDone. Plots saved: fig1_loss_and_refresh.png, "
          "fig2_wdm_capacity.png, fig3_banking_simulated.png")
