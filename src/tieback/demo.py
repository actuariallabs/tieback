#!/usr/bin/env python
"""Build the three-extract sample book: policies, vehicles and claims.

A pricing submission as it actually arrives, from three systems that were
never designed to be joined:

  policies.csv   policy admin: one row per policy term. POLICY_NO reads
                 'ON-0482113'; a renewal repeats it with the next term.
  vehicles.csv   vehicle schedule: one row per vehicle per term, with the
                 earned car-years (the exposure). POL_NUM is the bare number
                 482113, and the term is identified by its start date.
  claims.csv     claims system: one row per claim per coverage. POLICY_REF
                 reads 'P0482113'; there is no term, only a loss date, and a
                 claim with three coverages is three rows.

Planted, as in real extracts: claims on policies missing from the policy
extract, losses dated after a mid-term cancellation, policy terms with no
vehicle on the schedule, and a reported date next to the loss date (only
the loss date places a claim in its term).

The frequency signal is textbook (U-shaped driver age, region, at-fault
history, use, rate group, distance), so the joined book models sensibly.

Usage:  python -m tieback.demo [out_dir]
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("tieback_demo")

REGIONS = {  # weight, frequency relativity
    "Toronto": (0.22, 1.45), "GTA 905": (0.28, 1.25), "Hamilton-Niagara": (0.12, 1.05),
    "Ottawa": (0.12, 0.90), "Southwestern": (0.16, 0.85), "Northern": (0.10, 0.75),
}
USES = {"Pleasure": (0.45, 0.85), "Commute": (0.45, 1.00), "Business": (0.10, 1.30)}
AT_FAULT = {0: (0.82, 1.00), 1: (0.14, 1.35), 2: (0.035, 1.70), 3: (0.005, 2.10)}
COVERAGES = {  # weight as the first coverage, mean incurred, lognormal sigma
    "DCPD": (0.45, 4_500, 0.7), "COLL": (0.30, 7_000, 0.8), "BI": (0.10, 45_000, 1.1), "AB": (0.15, 18_000, 1.0),
}
BASE_FREQ = 0.065


def _pick(rng, table: dict, n: int):
    keys = list(table)
    w = np.array([table[k][0] for k in keys])
    return np.array(keys, dtype=object)[rng.choice(len(keys), n, p=w / w.sum())]


def build(out: Path = OUT, n_policies: int = 9_000, seed: int = 2027) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    numbers = rng.choice(np.arange(100_000, 1_000_000), n_policies + 40, replace=False)
    pol_nums, ghost_nums = numbers[:n_policies], numbers[n_policies:]

    region = _pick(rng, REGIONS, n_policies)
    age0 = np.clip(np.round(18 + rng.gamma(2.2, 11, n_policies)), 18, 88).astype(int)
    licensed0 = np.minimum(age0 - 16, np.round(rng.uniform(0.3, 1.0, n_policies) * (age0 - 16))).astype(int)
    at_fault = _pick(rng, AT_FAULT, n_policies).astype(int)
    home = np.where(rng.random(n_policies) < 0.45, "Y", "N")
    n_veh = rng.choice([1, 2, 3], n_policies, p=[0.60, 0.32, 0.08])
    inception = np.array([date(2023, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 365, n_policies)])

    pol_rows, veh_rows, terms = [], [], []
    for i in range(n_policies):
        units = [
            {"use": _pick(rng, USES, 1)[0], "group": int(rng.integers(1, 21)),
             "km": int(rng.choice(np.arange(5_000, 40_001, 1_000))), "year": int(rng.integers(2006, 2025))}
            for _ in range(n_veh[i])
        ]
        eff, term_no = inception[i], 1
        while True:
            cancelled = rng.random() < 0.08
            length = int(rng.integers(30, 330)) if cancelled else 365
            exp = eff + timedelta(days=length)
            terms.append((i, term_no, eff, exp, cancelled, units))
            pol_rows.append({
                "POLICY_NO": f"ON-{pol_nums[i]:07d}", "TERM_NO": term_no, "TERM_EFF": eff, "TERM_EXP": exp,
                "ORIG_INCEPTION": inception[i],
                "REGION": region[i], "DRIVER_AGE": int(age0[i] + term_no - 1),
                "YEARS_LICENSED": int(licensed0[i] + term_no - 1), "AT_FAULT_3YR": int(at_fault[i]), "HOME_POLICY": home[i],
            })
            if cancelled or term_no == 2 or rng.random() > 0.70:
                break
            eff, term_no = exp, term_no + 1

    claim_rows, claim_seq = [], 0

    def add_claim(ref: str, unit: int, loss: date):
        nonlocal claim_seq
        claim_seq += 1
        first = _pick(rng, COVERAGES, 1)[0]
        covs = [first]
        if first in ("BI", "COLL") and rng.random() < 0.4:
            covs.append("AB")
        if rng.random() < 0.2:
            covs.append("COLL" if first == "DCPD" else "DCPD")
        reported = loss + timedelta(days=int(rng.geometric(1 / 12)))
        for cov in dict.fromkeys(covs):
            _, mean, sigma = COVERAGES[cov]
            incurred = round(float(rng.lognormal(np.log(mean) - sigma**2 / 2, sigma)), 2)
            claim_rows.append({
                "CLAIM_NO": f"CLM-{claim_seq:07d}", "POLICY_REF": ref, "VEHICLE": unit, "COVERAGE": cov,
                "LOSS_DATE": loss, "REPORTED_DATE": reported,
                "INCURRED": incurred, "PAID": round(incurred * float(rng.uniform(0.3, 1.0)), 2),
            })

    for i, term_no, eff, exp, cancelled, units in terms:
        age = age0[i] + term_no - 1
        age_rel = 1 + 1.6 * np.exp(-(age - 18) / 6) + 0.012 * max(age - 65, 0)
        base = BASE_FREQ * REGIONS[region[i]][1] * age_rel * AT_FAULT[at_fault[i]][1] * (0.9 if home[i] == "Y" else 1.0)
        days = (exp - eff).days
        for u, unit in enumerate(units, start=1):
            expo = days / 365.25
            veh_rows.append({
                "POL_NUM": int(pol_nums[i]), "TERM_START": eff, "UNIT_NO": u, "VEH_YEAR": unit["year"],
                "VEH_USE": unit["use"], "RATE_GROUP": unit["group"], "ANNUAL_KM": unit["km"],
                "EARNED_CAR_YRS": round(expo, 4),
            })
            lam = base * USES[unit["use"]][1] * np.exp(0.03 * (unit["group"] - 10)) * (unit["km"] / 15_000) ** 0.3 * expo
            for _ in range(rng.poisson(lam)):
                add_claim(f"P{pol_nums[i]:07d}", u, eff + timedelta(days=int(rng.integers(0, days))))
        # Losses reported after a mid-term cancellation: no term covers them.
        if cancelled and rng.random() < 0.03:
            add_claim(f"P{pol_nums[i]:07d}", 1, exp + timedelta(days=int(rng.integers(5, 90))))

    # Claims on policies the policy extract does not contain.
    for n in ghost_nums[:30]:
        add_claim(f"P{n:07d}", 1, date(2024, 1, 1) + timedelta(days=int(rng.integers(0, 600))))
    # Policy terms with no vehicle on the schedule (dropped by the vehicle system).
    for j, n in enumerate(ghost_nums[30:]):
        eff = date(2024, 1, 1) + timedelta(days=int(rng.integers(0, 300)))
        pol_rows.append({
            "POLICY_NO": f"ON-{n:07d}", "TERM_NO": 1, "TERM_EFF": eff, "TERM_EXP": eff + timedelta(days=365),
            "ORIG_INCEPTION": eff, "REGION": "Toronto", "DRIVER_AGE": 40, "YEARS_LICENSED": 20,
            "AT_FAULT_3YR": 0, "HOME_POLICY": "N",
        })

    # Each system exports in its own order.
    frames = {
        "policies.csv": pl.DataFrame(pol_rows).sample(fraction=1.0, shuffle=True, seed=seed),
        "vehicles.csv": pl.DataFrame(veh_rows).sort("POL_NUM", "TERM_START", "UNIT_NO"),
        "claims.csv": pl.DataFrame(claim_rows).sort("REPORTED_DATE", "CLAIM_NO"),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.write_csv(out / name)
    return frames


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    for name, df in build(out).items():
        print(f"{name}: {df.height:,} rows x {df.width} cols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
