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


def build(out: Path = OUT, n_policies: int = 9_000, seed: int = 2027, *, messy: bool = False) -> dict[str, pl.DataFrame]:
    """Write the sample book. `messy=True` adds two more files (finance
    premium, driver convictions) and plants the problems real extracts have;
    `build_with_truth` also returns what was planted."""
    return build_with_truth(out, n_policies, seed, messy=messy)[0]


def build_with_truth(out: Path = OUT, n_policies: int = 9_000, seed: int = 2027, *,
                     messy: bool = False) -> tuple[dict[str, pl.DataFrame], dict]:
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
    truth: dict = {}
    if messy:
        frames, truth = _mess_up(frames, np.random.default_rng(seed + 1))
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.write_csv(out / name)
    return frames, truth


def _typo(rng, digits: str, kind: str, taken: set[str]) -> str | None:
    """A mistyped copy of a digit string that is not itself a real key."""
    for _ in range(50):
        if kind == "swap":
            i = int(rng.integers(0, len(digits) - 1))
            if digits[i] == digits[i + 1]:
                continue
            new = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
        else:
            i = int(rng.integers(1, len(digits)))
            new = digits[:i] + str((int(digits[i]) + int(rng.integers(1, 10))) % 10) + digits[i + 1:]
        if new != digits and new.lstrip("0") not in taken:
            return new
    return None


def _mess_up(frames: dict[str, pl.DataFrame], rng) -> tuple[dict[str, pl.DataFrame], dict]:
    """Two more files and the problems real extracts carry. Returns the files
    and a record of every problem planted, for testing."""
    pol, veh, clm = frames["policies.csv"], frames["vehicles.csv"], frames["claims.csv"]
    taken = set(pol["POLICY_NO"].str.slice(3).str.strip_chars_start("0").to_list())
    truth: dict = {}

    # Finance premium: one row per policy term, id '0482113/1', dates day first.
    prem = pol.select(
        (pl.col("POLICY_NO").str.slice(3) + "/" + pl.col("TERM_NO").cast(pl.Utf8)).alias("GL_POLICY_TERM"),
        pl.col("TERM_EFF").dt.strftime("%d/%m/%Y").alias("BOOKED_FROM"),
        pl.Series("WRITTEN_PREMIUM", np.round(rng.uniform(700, 3_200, pol.height), 2)),
    ).sample(fraction=1.0, shuffle=True, seed=1)
    # Driver convictions: per policy, dated, several per policy term.
    pick = rng.integers(0, pol.height, max(pol.height // 5, 50))
    eff = pol["TERM_EFF"].to_list()
    days = (pol["TERM_EXP"] - pol["TERM_EFF"]).dt.total_days().to_list()
    conv = pl.DataFrame({
        "CONVICTION_ID": [f"MTO-{i:06d}" for i in range(len(pick))],
        "POLICYNUM": [pol["POLICY_NO"][int(i)][3:] for i in pick],
        "CONV_DATE": [eff[int(i)] + timedelta(days=int(rng.integers(0, max(int(days[int(i)]), 1)))) for i in pick],
        "OFFENCE": rng.choice(["Speeding", "Distracted driving", "Careless driving", "Fail to stop"], len(pick)),
    })

    # Mistyped policy numbers on claims: neighbours swapped (applied by
    # default) and one digit changed (reported as likely).
    claim_ids = clm.filter(pl.col("POLICY_REF").str.slice(1).str.strip_chars_start("0").is_in(list(taken)))["CLAIM_NO"].unique().sort()
    chosen = rng.choice(claim_ids.to_list(), 12, replace=False)
    truth["claim_swaps"], truth["claim_changes"] = {}, {}
    fix = {}
    for j, cid in enumerate(chosen):
        ref = clm.filter(pl.col("CLAIM_NO") == cid)["POLICY_REF"][0]
        kind = "swap" if j < 8 else "change"
        new = _typo(rng, ref[1:], kind, taken)
        if new:
            fix[cid] = "P" + new
            truth["claim_swaps" if kind == "swap" else "claim_changes"][cid] = (ref, "P" + new)
    clm = clm.with_columns(pl.col("CLAIM_NO").replace_strict(fix, default=pl.col("POLICY_REF")).alias("POLICY_REF"))

    # A mistyped policy number on the vehicle schedule (the exposure file).
    vrows = rng.choice(veh.height, 3, replace=False)
    truth["vehicle_swaps"] = {}
    nums = veh["POL_NUM"].to_list()
    for r in vrows:
        new = _typo(rng, f"{nums[r]}", "swap", taken)
        if new:
            truth["vehicle_swaps"][int(r)] = (nums[r], int(new))
            nums[r] = int(new)
    veh = veh.with_columns(pl.Series("POL_NUM", nums))

    # Exposure longer than its term.
    over = rng.choice(veh.height, 2, replace=False)
    expo = veh["EARNED_CAR_YRS"].to_list()
    for r in over:
        expo[r] = round(expo[r] + 0.75, 4)
    veh = veh.with_columns(pl.Series("EARNED_CAR_YRS", expo))
    truth["exposure_over"] = 2

    # One region spelt three ways.
    reg = pol["REGION"].to_list()
    idx = [i for i, r in enumerate(reg) if r == "Toronto"]
    for i in idx[:20]:
        reg[i] = "TORONTO"
    for i in [i for i, r in enumerate(reg) if r == "Ottawa"][:5]:
        reg[i] = "Otawa"
    pol = pol.with_columns(pl.Series("REGION", reg))
    truth["region_variants"] = {"TORONTO": "Toronto", "Otawa": "Ottawa"}

    # Claim amounts: a sign error, paid above incurred, dates swapped.
    rows = rng.choice(clm.height, 9, replace=False)
    inc, paid = clm["INCURRED"].to_list(), clm["PAID"].to_list()
    loss, rep = clm["LOSS_DATE"].to_list(), clm["REPORTED_DATE"].to_list()
    for r in rows[:2]:
        inc[r], paid[r] = -inc[r], 0.0
    for r in rows[2:6]:
        paid[r] = round(inc[r] * 1.3, 2)
    for r in rows[6:9]:
        loss[r], rep[r] = rep[r] + timedelta(days=3), loss[r]
    clm = clm.with_columns(pl.Series("INCURRED", inc), pl.Series("PAID", paid),
                           pl.Series("LOSS_DATE", loss), pl.Series("REPORTED_DATE", rep))
    truth.update(negative=2, paid_over=4, reported_before=3)

    # Rows exported twice.
    veh = pl.concat([veh, veh[rng.choice(veh.height, 3, replace=False)]]).sort("POL_NUM", "TERM_START", "UNIT_NO")
    clm = pl.concat([clm, clm[rng.choice(clm.height, 4, replace=False)]]).sort("REPORTED_DATE", "CLAIM_NO")
    truth.update(dup_vehicles=3, dup_claims=4)

    return ({"policies.csv": pol, "vehicles.csv": veh, "claims.csv": clm,
             "premium.csv": prem, "convictions.csv": conv}, truth)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    for name, df in build(out, messy=True).items():
        print(f"{name}: {df.height:,} rows x {df.width} cols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
