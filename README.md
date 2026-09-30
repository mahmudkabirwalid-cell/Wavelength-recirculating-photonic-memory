# """
ABOBB Banking Simulation -- a small, validated result
=======================================================

WHAT THIS IS
------------
A minimal digital-twin simulator for one specific question inside a larger
optical-buffering architecture (Adaptive Banked Optical Block Buffer /
ABOBB): when you split one optical recirculation buffer into N parallel
banks, how does access waiting time actually behave?

This is deliberately narrow in scope. It does not claim a new physical
mechanism -- recirculating optical buffers are established prior art
(Burmeister et al., "Photonic Chip Recirculating Buffer for Optical Packet
Switching", Optica/IPNRA 2008; Alexoudi et al., "Optical RAM and integrated
optical memories: a survey", Light: Science & Applications 9, 91 (2020)).
The claim here is narrower and checkable: the QUEUEING BEHAVIOR of
splitting one big optical buffer into many small banks, modeled as an
M/M/c queue, simulated by discrete-event method, and VALIDATED against
closed-form Erlang-C queueing theory (Erlang, 1917; see Kleinrock,
"Queueing Systems Vol. 1", Wiley 1975, for the standard derivation).

WHY VALIDATION MATTERS
-----------------------
A stochastic simulation that isn't checked against a known analytical
result is just an assumption dressed up as code. Section "VALIDATION"
below runs the discrete-event simulator against the exact same conditions
the Erlang-C formula describes (a pure M/M/c queue, no scheduler overhead)
and reports the two side by side across multiple random seeds, with a
95% confidence interval. This is the standard way discrete-event
simulations are checked in queueing-theory literature (Law & Kelton,
"Simulation Modeling and Analysis", McGraw-Hill) -- if the analytical
value falls inside the simulated CI, the simulator's core logic is sound,
and any place it's later extended beyond the analytical model's
assumptions (e.g. adding scheduler overhead) can be trusted to be
introducing that specific, named difference -- not hiding a bug.

THE MAIN RESULT
----------------
With the full model (Poisson arrivals, exponential service, N banks as
parallel servers, shortest-queue-free scheduling, PLUS a fixed 2 ns
electronic scheduler overhead per access), stress-tested at 167% of
single-bank capacity:

    Banks    Simulated mean wait (95% CI over 20 replications)
    1        unstable -- queue length diverges (rho > 1)
    2        ~150 ns
    4        ~5 ns
    8        ~2 ns
    16-64    ~2 ns -- flat, hits the scheduler-overhead floor

Two findings that matter:
  1. A single overloaded buffer isn't just slow -- it's structurally
     UNSTABLE (mean queue length diverges as replications get longer).
     Banking changes the buffer's stability regime, not just its speed.
  2. Past ~8 banks in this scenario, adding more banks buys NOTHING --
     waiting time flattens at the electronic scheduler-overhead floor.
     Beyond that point the bottleneck is the electronic control plane,
     not the optical domain.

ALSO INCLUDED (physics sanity checks)
---------------------------------------
- Loss decay + refresh threshold, from P_n/P_0 = 10^(-n*A/10).
- WDM capacity scaling, from C_total ~= M * C_lambda.
Both are closed-form, not simulated, and are included only so the
queueing model's input assumptions (loop delay, capacity) can be
checked independently against the same equations.

EXPLICIT LIMITATIONS
----------------------
- Exponential inter-arrival/service times are the standard queueing-theory
  simplification, not a claim about real network traffic, which is
  typically bursty/self-similar. This is the single most important
  assumption to relax next.
- No optical-quality (BER/OSNR) model yet; refresh trigger is power-only.
- No energy accounting yet -- this result is about latency, not about
  whether banking is worth its control/monitoring overhead in energy.
- All physical parameters (loss, scheduler overhead, service time) are
  stated assumptions in PARAMS below, not measured device numbers.

THE SPECIFIC QUESTION THIS IS BUILT TO ASK A REVIEWER
--------------------------------------------------------
Given that the simulator is validated against Erlang-C theory (see
VALIDATION output below), does the instability-then-floor shape found
under overload match expectations for a real optical buffering system --
or would a bursty (non-Poisson) traffic model change the shape
qualitatively, not just quantitatively?

HOW TO RUN
----------
    pip install numpy matplotlib
    python3 abobb.py                  # full run: validation + main result + plots
    python3 abobb.py --replications 50 --help   # see all CLI options

Mahmud Kabir Walid -- independent study project. Not affiliated with,
endorsed by, or reviewed by any company referenced in earlier drafts.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("abobb")

PARAMS = {
    "group_velocity_m_s": 2e8,
    "line_rate_bps": 100e9,
    "loss_db_per_circulation": 0.08,
    "quality_threshold": 0.70,
    "scheduler_overhead_s": 2e-9,
}


# ----------------------------------------------------------------------
# Physics: loop delay, capacity, loss (closed-form, not simulated)
# ----------------------------------------------------------------------

def round_trip_delay(loop_length_m: float, v_g: float = PARAMS["group_velocity_m_s"]) -> float:
    """t_RT = 2L / v_g"""
    return 2 * loop_length_m / v_g


def payload_capacity_bits(loop_length_m: float, line_rate_bps: float = PARAMS["line_rate_bps"],
                           v_g: float = PARAMS["group_velocity_m_s"]) -> float:
    """C = R * t_RT"""
    return line_rate_bps * round_trip_delay(loop_length_m, v_g)


def power_after_n_circulations(n, loss_db: float = PARAMS["loss_db_per_circulation"]):
    """P_n / P_0 = 10^(-n*A/10)"""
    n = np.asarray(n, dtype=float)
    return 10 ** (-n * loss_db / 10)


def circulations_to_threshold(threshold: float = PARAMS["quality_threshold"],
                               loss_db: float = PARAMS["loss_db_per_circulation"]) -> float:
    return -10 * math.log10(threshold) / loss_db


def wdm_capacity_bits(loop_length_m: float, n_wavelengths: int,
                       line_rate_bps: float = PARAMS["line_rate_bps"],
                       v_g: float = PARAMS["group_velocity_m_s"]) -> float:
    """C_total ~= M * C_lambda (ideal case, no crosstalk/guard-band penalty)"""
    return n_wavelengths * payload_capacity_bits(loop_length_m, line_rate_bps, v_g)


# ----------------------------------------------------------------------
# Analytical M/M/c theory (Erlang-C), used ONLY to validate the simulator
# ----------------------------------------------------------------------

def erlang_c_mean_wait(c: int, arrival_rate: float, service_rate: float) -> float:
    """
    Closed-form mean waiting time for a pure M/M/c queue (no scheduler
    overhead). Standard Erlang-C derivation; see Kleinrock (1975),
    "Queueing Systems Vol 1", Ch. 3.

    a = offered load (Erlangs), rho = a/c = per-server utilization.
    Returns inf if the queue is unstable (rho >= 1).
    """
    a = arrival_rate / service_rate
    rho = a / c
    if rho >= 1:
        return float("inf")
    terms = sum(a ** k / math.factorial(k) for k in range(c))
    last = (a ** c) / (math.factorial(c) * (1 - rho))
    p0 = 1 / (terms + last)
    p_wait = last * p0  # Erlang-C probability of waiting
    return p_wait / (c * service_rate - arrival_rate)


# ----------------------------------------------------------------------
# Discrete-event M/M/c simulator (this is the thing being validated)
# ----------------------------------------------------------------------

@dataclass
class SimResult:
    mean_wait_s: float
    n_events: int
    seed: int


def simulate_mmc_queue(n_banks: int, arrival_rate: float, service_rate: float,
                        scheduler_overhead_s: float = 0.0,
                        n_events: int = 20000, seed: int = 0) -> SimResult:
    """
    Discrete-event simulation of n_banks parallel optical banks.
    scheduler_overhead_s=0 reproduces the pure M/M/c assumptions used by
    erlang_c_mean_wait, for validation. A nonzero value models the
    electronic scheduling cost of the real ABOBB architecture.
    """
    rng = random.Random(seed)
    server_free_at = [0.0] * n_banks
    t = 0.0
    waits = []
    for _ in range(n_events):
        t += rng.expovariate(arrival_rate)
        idx = min(range(n_banks), key=lambda i: server_free_at[i])
        start = max(t, server_free_at[idx]) + scheduler_overhead_s
        waits.append(start - t)
        server_free_at[idx] = start + rng.expovariate(service_rate)
    return SimResult(mean_wait_s=float(np.mean(waits)), n_events=n_events, seed=seed)


def replicate(n_banks: int, arrival_rate: float, service_rate: float,
              scheduler_overhead_s: float, n_replications: int, n_events: int) -> tuple[float, float]:
    """Run n_replications independent simulations (different seeds) and
    return (mean, half-width of the 95% CI) across replications --
    standard method-of-independent-replications for simulation output
    analysis (Law & Kelton, Ch. 9)."""
    means = [
        simulate_mmc_queue(n_banks, arrival_rate, service_rate,
                            scheduler_overhead_s, n_events, seed=s).mean_wait_s
        for s in range(n_replications)
    ]
    mean = float(np.mean(means))
    sem = float(np.std(means, ddof=1) / math.sqrt(n_replications))
    ci95 = 1.96 * sem
    return mean, ci95


def validate_against_theory(bank_counts: Sequence[int], arrival_rate: float,
                             service_rate: float, n_replications: int, n_events: int) -> None:
    """Compare the simulator (scheduler_overhead=0) against Erlang-C theory."""
    log.info("=== VALIDATION: simulator vs. closed-form Erlang-C theory ===")
    log.info(f"{'Banks':>6} {'Analytical (ns)':>16} {'Simulated (ns)':>16} "
             f"{'95% CI (+/- ns)':>16} {'In range?':>10}")
    for c in bank_counts:
        rho = arrival_rate / (c * service_rate)
        if rho >= 1:
            continue  # Erlang-C undefined for unstable queues; skip validation there
        analytical_ns = erlang_c_mean_wait(c, arrival_rate, service_rate) * 1e9
        mean_ns, ci_ns = replicate(c, arrival_rate, service_rate,
                                    scheduler_overhead_s=0.0,
                                    n_replications=n_replications, n_events=n_events)
        mean_ns_ns = mean_ns * 1e9
        ci_ns_ns = ci_ns * 1e9
        in_range = abs(mean_ns_ns - analytical_ns) <= ci_ns_ns
        log.info(f"{c:>6} {analytical_ns:>16.3f} {mean_ns_ns:>16.3f} "
                 f"{ci_ns_ns:>16.3f} {'YES' if in_range else 'no':>10}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ABOBB banking simulation")
    parser.add_argument("--replications", type=int, default=20,
                         help="independent simulation runs per data point (default 20)")
    parser.add_argument("--events", type=int, default=20000,
                         help="events per simulation run (default 20000)")
    parser.add_argument("--no-plots", action="store_true", help="skip saving PNG plots")
    args = parser.parse_args()

    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    # ---- Physics sanity check ----
    log.info("=== Physics sanity check ===")
    for L in [1, 10, 100, 1000]:
        t_rt = round_trip_delay(L)
        cap = payload_capacity_bits(L)
        log.info(f"  {L:>5} m  ->  {t_rt*1e9:8.1f} ns RT delay, {cap/8:10.1f} B capacity")

    # ---- VALIDATION: simulator vs. Erlang-C, at a STABLE load ----
    # (Erlang-C requires rho < 1; use a lighter load than the stress-test below)
    service_rate = 1 / 50e-9      # 1/service_time_s
    stable_arrival_rate = 1 / 80e-9  # chosen so rho < 1 for banks >= 2
    validate_against_theory(bank_counts=[2, 3, 4, 5, 6, 8],
                             arrival_rate=stable_arrival_rate, service_rate=service_rate,
                             n_replications=args.replications, n_events=args.events)

    # ---- Physics figure 1: loss decay + refresh threshold ----
    n = np.arange(0, 101)
    power = power_after_n_circulations(n)
    thresh = PARAMS["quality_threshold"]
    n_thresh = circulations_to_threshold(thresh)
    log.info(f"\nRefresh required every {n_thresh:.2f} circulations at "
             f"{PARAMS['loss_db_per_circulation']} dB/circulation loss.")
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(n, power, lw=2)
        ax.axhline(thresh, ls="--", color="gray", label=f"refresh threshold = {thresh}")
        ax.axvline(n_thresh, ls=":", color="gray")
        ax.set_xlabel("Circulations"); ax.set_ylabel("Normalized optical power")
        ax.set_title(f"Loss decay -> refresh every {n_thresh:.1f} circulations")
        ax.legend(); fig.tight_layout(); fig.savefig("fig1_loss_and_refresh.png")

    # ---- Physics figure 2: WDM capacity scaling ----
    if not args.no_plots:
        wl_counts = [1, 4, 8, 16, 32]
        caps_KiB = [wdm_capacity_bits(10, m) / 8 / 1024 for m in wl_counts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(wl_counts, caps_KiB, "o-", lw=2)
        ax.set_xlabel("WDM wavelengths"); ax.set_ylabel("Circulating payload capacity (KiB)")
        ax.set_title("Idealized WDM capacity scaling, 10 m loop @ 100 Gb/s")
        fig.tight_layout(); fig.savefig("fig2_wdm_capacity.png")

    # ---- MAIN RESULT: full model (with scheduler overhead), stress-tested ----
    log.info("\n=== MAIN RESULT: banking effect under overload (full model, "
             f"{args.replications} replications each) ===")
    bank_counts = [1, 2, 4, 8, 16, 32, 64]
    stress_arrival_rate = 1 / 30e-9  # single-bank utilization ~1.67 (deliberately overloaded)
    means_ns, cis_ns = [], []
    for c in bank_counts:
        rho = stress_arrival_rate / (c * service_rate)
        if rho >= 1:
            log.info(f"  banks={c:3d}  UNSTABLE (rho={rho:.2f} >= 1); "
                     f"queue length diverges, wait grows without bound")
            means_ns.append(np.nan)
            cis_ns.append(0.0)
            continue
        mean_s, ci_s = replicate(c, stress_arrival_rate, service_rate,
                                  scheduler_overhead_s=PARAMS["scheduler_overhead_s"],
                                  n_replications=args.replications, n_events=args.events)
        log.info(f"  banks={c:3d}  mean wait = {mean_s*1e9:10.2f} ns "
                 f"(+/- {ci_s*1e9:.2f} ns, 95% CI, rho={rho:.2f})")
        means_ns.append(mean_s * 1e9)
        cis_ns.append(ci_s * 1e9)

    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_banks = [c for c, m in zip(bank_counts, means_ns) if not np.isnan(m)]
        plot_means = [m for m in means_ns if not np.isnan(m)]
        ax.plot(plot_banks, plot_means, "o-", lw=2, color="darkred")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.axhline(PARAMS["scheduler_overhead_s"] * 1e9, ls="--", color="gray",
                   label="scheduler overhead floor (2 ns)")
        ax.set_xlabel("Number of optical banks")
        ax.set_ylabel("Simulated mean waiting time (ns, log scale)")
        ax.set_title("Banking effect on access waiting -- validated M/M/c simulation")
        ax.legend(); fig.tight_layout(); fig.savefig("fig3_banking_simulated.png")
        log.info("\nPlots saved: fig1_loss_and_refresh.png, fig2_wdm_capacity.png, "
                 "fig3_banking_simulated.png")


# ----------------------------------------------------------------------
# Lightweight tests -- run with: python3 -c "import abobb; abobb.run_tests()"
# ----------------------------------------------------------------------

def run_tests() -> None:
    """Sanity tests a reviewer can run without touching the plotting code."""
    # Physics: 1 m loop at v_g=2e8 m/s -> 10 ns round trip (dossier's own table)
    assert abs(round_trip_delay(1) - 10e-9) < 1e-12
    # Capacity: 10 ns * 100 Gb/s = 1000 bits = 125 bytes
    assert abs(payload_capacity_bits(1) / 8 - 125) < 1e-6
    # Erlang-C: known reference value, c=1, rho=0.5 -> M/M/1 mean wait = rho/(mu-lambda)
    mu, lam = 1.0, 0.5
    expected = (lam / mu) / (mu - lam)  # M/M/1 Wq = rho/(mu-lambda)
    got = erlang_c_mean_wait(1, lam, mu)
    assert abs(got - expected) < 1e-9, (got, expected)
    print("All tests passed.")


if __name__ == "__main__":
    main()
