"""Five extracts with the problems real ones have, checked against what was
planted: mistyped keys, a two-number id, files at a coarser grain, repeated
rows, inflated exposure, bad amounts and dates, and misspelt categories."""

from __future__ import annotations

import polars as pl
import pytest

import tieback
from tieback.demo import build_with_truth
from tieback.repair import _distance_one, rewrite


@pytest.fixture(scope="module")
def messy(tmp_path_factory):
    out = tmp_path_factory.mktemp("messy")
    frames, truth = build_with_truth(out, n_policies=3_000, seed=2027, messy=True)
    tables = tieback.read(sorted(out.glob("*.csv")))
    book, plan = tieback.assemble(tables)
    return tables, truth, book, plan


def by_check(plan, check):
    return [f for f in plan.findings if f["check"] == check]


class TestJoins:
    def test_every_file_joins_the_right_way(self, messy):
        _, _, _, plan = messy
        assert not plan.unresolved
        kinds = {s.table: s.kind for s in plan.steps}
        assert kinds == {"policies.csv": "lookup", "claims.csv": "rollup",
                         "premium.csv": "group", "convictions.csv": "group"}

    def test_two_number_id_is_split(self, messy):
        _, _, _, plan = messy
        prem = next(s for s in plan.steps if s.table == "premium.csv")
        assert prem.keys[0].other == "GL_POLICY_TERM·1"

    def test_part_columns_never_reach_the_table(self, messy):
        _, _, book, _ = messy
        assert not [c for c in book.columns if "·" in c]

    def test_rows_and_exposure_unchanged(self, messy):
        tables, _, book, plan = messy
        assert book.height == tables["vehicles.csv"].height
        assert plan.totals["exposure_in"] == pytest.approx(plan.totals["exposure_out"])

    def test_premium_is_split_by_exposure_and_still_ties(self, messy):
        tables, _, book, plan = messy
        prem = next(s for s in plan.steps if s.table == "premium.csv")
        a = prem.totals["amounts"]["WRITTEN_PREMIUM"]
        assert a["in_file"] == pytest.approx(a["in_table"] + a["no_key"] + a["outside_window"] + a["ambiguous"])
        assert book["WRITTEN_PREMIUM_sum"].sum() == pytest.approx(a["attached"])
        # A one-vehicle term carries its whole premium; nothing is double counted.
        assert book["WRITTEN_PREMIUM_sum"].sum() < tables["premium.csv"]["WRITTEN_PREMIUM"].sum() + 1e-6

    def test_convictions_count_is_per_policy_term(self, messy):
        _, _, book, plan = messy
        conv = next(s for s in plan.steps if s.table == "convictions.csv")
        assert conv.window is not None and conv.window.date == "CONV_DATE"
        same = book.group_by("POL_NUM", "TERM_START").agg(pl.col("convictions_count").n_unique().alias("n"))
        assert same["n"].max() == 1


class TestKeys:
    def test_vehicle_typos_are_corrected(self, messy):
        _, truth, _, plan = messy
        fixes = {r["original"]: r["corrected"] for s in plan.steps for r in s.repairs}
        for orig, typo in truth["vehicle_swaps"].values():
            assert fixes.get(str(typo)) == str(orig)

    def test_claim_swaps_are_corrected_and_corrections_are_right(self, messy):
        _, truth, _, plan = messy
        fixes = {r["original"]: r["corrected"] for s in plan.steps for r in s.repairs if r["table"] == "claims.csv"}
        planted = {typo: orig for orig, typo in truth["claim_swaps"].values()}
        right = sum(fixes.get(t) == o for t, o in planted.items())
        wrong = sum(1 for t, c in fixes.items() if planted.get(t) != c)
        assert right >= len(planted) // 2
        assert wrong <= 1

    def test_single_digit_changes_are_reported_not_applied(self, messy):
        _, truth, _, plan = messy
        fixes = {r["original"] for s in plan.steps for r in s.repairs}
        flagged = {sp["value"] for sp in plan.suspects}
        for orig, typo in truth["claim_changes"].values():
            assert typo not in fixes or typo in flagged

    def test_chance_check_is_reported(self, messy):
        _, _, _, plan = messy
        claims = next(s for s in plan.steps if s.table == "claims.csv")
        assert claims.repair_stats["unmatched"] > 0
        assert "chance_rate" in claims.repair_stats

    def test_repair_off_changes_nothing(self, messy):
        tables, _, _, _ = messy
        _, plan = tieback.assemble(tables, repair=False)
        assert not any(s.repairs for s in plan.steps)

    def test_saved_plan_replays_with_its_corrections(self, messy, tmp_path):
        tables, _, _, plan = messy
        tieback.save_plan(plan, tmp_path / "plan.json")
        result = tieback.replay(tables, tieback.load_plan(tmp_path / "plan.json"))
        assert {s["table"] for s in result["steps"]} == {s.table for s in plan.steps}


class TestFindings:
    def test_repeated_rows(self, messy):
        _, truth, _, plan = messy
        found = {f["table"]: f["count"] for f in by_check(plan, "duplicate_rows")}
        assert found["claims.csv"] == truth["dup_claims"]
        assert found["vehicles.csv"] == truth["dup_vehicles"]

    def test_double_exposure(self, messy):
        _, truth, _, plan = messy
        assert by_check(plan, "overlapping_periods")[0]["count"] >= truth["dup_vehicles"]
        assert by_check(plan, "exposure_exceeds_period")[0]["count"] == truth["exposure_over"]

    def test_amounts_and_dates(self, messy):
        _, truth, _, plan = messy
        assert by_check(plan, "negative_amount")[0]["count"] >= truth["negative"]
        assert by_check(plan, "paid_exceeds_incurred")[0]["count"] >= truth["paid_over"]
        assert by_check(plan, "reported_before_loss")[0]["count"] >= truth["reported_before"]

    def test_misspelt_categories(self, messy):
        _, truth, _, plan = messy
        f = next(f for f in by_check(plan, "category_spellings") if f["column"] == "REGION")
        assert {e["from"]: e["to"] for e in f["examples"]} == truth["region_variants"]

    def test_control_totals(self, messy):
        tables, _, book, _ = messy
        total = float(book["EARNED_CAR_YRS"].sum())
        _, plan = tieback.assemble(tables, controls={"EARNED_CAR_YRS": total * 1.02})
        f = by_check(plan, "control_total")[0]
        assert f["severity"] == "error" and "does not tie" in f["message"]

    def test_errors_come_first_and_report_prints(self, messy):
        _, _, _, plan = messy
        sev = [f["severity"] for f in plan.findings]
        assert sev == sorted(sev, key=["error", "warning", "info"].index)
        text = tieback.report(plan)
        assert "findings:" in text and "·" not in text


class TestCleanBook:
    def test_no_corrections_and_no_errors_on_clean_extracts(self, tmp_path):
        frames, _ = build_with_truth(tmp_path, n_policies=3_000, seed=2027)
        book, plan = tieback.assemble(tieback.read(sorted(tmp_path.glob("*.csv"))))
        assert not [r for s in plan.steps for r in s.repairs]
        assert not [f for f in plan.findings if f["severity"] == "error"]


class TestPieces:
    def test_slip_kinds(self):
        assert _distance_one("412403", "412430") == "swapped neighbours"
        assert _distance_one("412403", "412408") == "one character"
        assert _distance_one("412403", "41243") == "one character"
        assert _distance_one("412403", "421430") is None

    def test_rewrite_keeps_the_format(self):
        assert rewrite("P0412430", "digits", "412403") == "P0412403"
        assert rewrite("ab-1003", "exact", "AB-1003") == "AB-1003"

    def test_case_and_spacing_on_text_ids(self):
        import random
        rng = random.Random(3)
        ids = sorted({"".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8)) for _ in range(200)})
        base = pl.DataFrame({"BROKER_CODE": ids, "EXPOSURE": [1.0] * len(ids)})
        refs = [ids[7].lower() + " ", ids[9]] + ids[20:60]
        claims = pl.DataFrame({"BROKER": refs, "CLAIM_ID": [f"C{i}" for i in range(len(refs))], "PAID": [100.0] * len(refs)})
        book, plan = tieback.assemble({"brokers.csv": base, "claims.csv": claims})
        fixes = {r["original"]: (r["corrected"], r["kind"]) for s in plan.steps for r in s.repairs}
        assert fixes == {ids[7].lower() + " ": (ids[7], "case and spacing")}
        assert book["claim_count"].sum() == len(refs)
