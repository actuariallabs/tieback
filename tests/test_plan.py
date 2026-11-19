"""Three extracts into one modelling table: the join plan.

Built on the generated Ontario book (policies by term, vehicles carrying the
exposure, claims by coverage), and checked against an independent join on
the keys the generator actually used, not against the plan's own report.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tieback.demo import build
from tieback.plan import assemble, parse_dates
from tieback.keys import detect_join_keys, normalize_key



@pytest.fixture(scope="module")
def book(tmp_path_factory) -> dict[str, pl.DataFrame]:
    out = tmp_path_factory.mktemp("book")
    build(out, n_policies=1_500, seed=11)
    # Read back from CSV, as an upload arrives: dates as text, ids as inferred.
    return {n: pl.read_csv(out / n) for n in ("policies.csv", "vehicles.csv", "claims.csv")}


@pytest.fixture(scope="module")
def assembled(book):
    return assemble(book)


def _truth(book, out: pl.DataFrame) -> pl.DataFrame:
    """Claims per modelling row, joined on the generator's own keys."""
    claims = parse_dates(book["claims.csv"]).with_columns(
        pl.col("POLICY_REF").str.slice(1).cast(pl.Int64).alias("POL_NUM"))
    rows = out.with_row_index("rid").select("rid", "POL_NUM", "UNIT_NO", "TERM_START", "TERM_EXP")
    hit = (claims.join(rows, left_on=["POL_NUM", "VEHICLE"], right_on=["POL_NUM", "UNIT_NO"])
                 .filter((pl.col("LOSS_DATE") >= pl.col("TERM_START")) & (pl.col("LOSS_DATE") < pl.col("TERM_EXP"))))
    per = hit.group_by("rid").agg(pl.col("CLAIM_NO").n_unique().alias("claims"), pl.col("INCURRED").sum().alias("incurred"))
    return out.with_row_index("rid").join(per, on="rid", how="left").with_columns(
        pl.col("claims").fill_null(0), pl.col("incurred").fill_null(0.0))


class TestPlan:
    def test_base_is_the_table_with_exposure(self, assembled):
        _, plan = assembled
        assert plan.base == "vehicles.csv"
        assert plan.exposure == "EARNED_CAR_YRS"

    def test_policies_look_up_onto_vehicles_by_number_and_term(self, assembled):
        _, plan = assembled
        step = plan.steps[0]
        assert (step.table, step.kind) == ("policies.csv", "lookup")
        first, term = step.keys
        # 'ON-0482113' in the policy system is 482113 on the vehicle schedule.
        assert (first.base, first.other, first.form) == ("POL_NUM", "POLICY_NO", "digits")
        assert first.example[1].startswith("ON-0")
        # Renewals repeat the number: the term start makes it one row.
        assert (term.base, term.other) == ("TERM_START", "TERM_EFF")

    def test_claims_roll_up_by_vehicle_and_loss_date(self, assembled):
        _, plan = assembled
        step = plan.steps[1]
        assert (step.table, step.kind) == ("claims.csv", "rollup")
        assert [(k.base, k.other) for k in step.keys] == [("POL_NUM", "POLICY_REF"), ("UNIT_NO", "VEHICLE")]
        # The loss date places a claim in its term; the reported date, which
        # can fall after the term ends, places fewer and is not chosen.
        assert (step.window.start, step.window.end, step.window.date) == ("TERM_START", "TERM_EXP", "LOSS_DATE")
        assert any("REPORTED_DATE" in r for r in step.reasons)
        # One row per coverage: claims are counted by claim number.
        assert step.count_of == "CLAIM_NO"
        assert step.sums == ["INCURRED", "PAID"]
        assert plan.unresolved == []


class TestControlTotals:
    def test_rows_and_exposure_unchanged(self, book, assembled):
        out, plan = assembled
        assert out.height == book["vehicles.csv"].height
        assert out["EARNED_CAR_YRS"].sum() == pytest.approx(book["vehicles.csv"]["EARNED_CAR_YRS"].sum(), abs=1e-6)
        assert plan.totals["exposure_out"] == pytest.approx(plan.totals["exposure_in"], abs=1e-6)

    def test_every_claim_is_attached_or_explained(self, book, assembled):
        _, plan = assembled
        t = plan.steps[1].totals
        assert t["units_in_file"] == book["claims.csv"]["CLAIM_NO"].n_unique()
        assert t["units_attached"] + t["units_no_key"] + t["units_outside_window"] + t["units_ambiguous"] == t["units_in_file"]
        # The generator plants claims on 30 unknown policies and losses after
        # a cancellation.
        assert t["units_no_key"] == 30
        assert t["units_outside_window"] > 0
        assert t["units_ambiguous"] == 0
        assert t["units_in_table"] == t["units_attached"]

    def test_every_dollar_reconciles(self, assembled):
        _, plan = assembled
        for amounts in plan.steps[1].totals["amounts"].values():
            parts = amounts["attached"] + amounts["no_key"] + amounts["outside_window"] + amounts["ambiguous"]
            assert parts == pytest.approx(amounts["in_file"], abs=0.01)
            assert amounts["in_table"] == pytest.approx(amounts["attached"], abs=0.01)

    def test_counts_match_an_independent_join(self, book, assembled):
        out, _ = assembled
        truth = _truth(book, out)
        assert (truth["claims"] != truth["claim_count"]).sum() == 0
        assert (truth["incurred"] - truth["INCURRED_sum"]).abs().max() < 0.01

    def test_lookup_reports_unused_policy_terms(self, assembled):
        _, plan = assembled
        t = plan.steps[0].totals
        assert t["base_rows_unmatched"] == 0
        assert t["other_rows_unused"] == 10  # terms with no vehicle on the schedule


class TestShapes:
    def test_two_files_one_key_is_a_rollup(self):
        policies = pl.DataFrame({"policy_id": range(500), "exposure": [0.5] * 500, "age": [30 + i % 40 for i in range(500)]})
        claims = pl.DataFrame({"policy_id": [i * 7 % 500 for i in range(120)], "paid_amount": [100.0] * 120})
        out, plan = assemble({"pol.csv": policies, "clm.csv": claims})
        assert [s.kind for s in plan.steps] == ["rollup"]
        assert out.height == 500 and out["claim_count"].sum() == 120
        assert out["paid_amount_sum"].sum() == pytest.approx(12_000.0)

    def test_table_with_no_shared_key_is_reported_not_dropped(self):
        policies = pl.DataFrame({"policy_id": range(300), "exposure": [1.0] * 300})
        claims = pl.DataFrame({"policy_id": range(0, 600, 2), "paid": [1.0] * 300})
        other = pl.DataFrame({"code": [f"Z{i}" for i in range(300)]})
        _, plan = assemble({"pol.csv": policies, "clm.csv": claims, "misc.csv": other})
        assert [u["table"] for u in plan.unresolved] == ["misc.csv"]
        # Half the policies have a claim and none has two: still events, by
        # their paid amounts, never policy attributes.
        assert [(s.table, s.kind) for s in plan.steps] == [("clm.csv", "rollup")]

    def test_dates_parse_from_text(self):
        df = parse_dates(pl.DataFrame({"d": ["2024-01-31", "2024-02-29", None], "x": ["a", "b", "c"]}))
        assert df["d"].dtype == pl.Date and df["d"][1] == date(2024, 2, 29)
        assert df["x"].dtype == pl.Utf8


class TestKeyForms:
    def test_prefix_and_padding_drop_to_the_number(self):
        s = pl.Series(["ON-0482113", "ON-0000007", None])
        assert normalize_key(s, "digits").to_list() == ["482113", "7", None]

    def test_values_with_several_numbers_have_no_digits_form(self):
        assert normalize_key(pl.Series(["2024-01-31", "2024-02-01"]), "digits") is None
        assert normalize_key(pl.Series(["CLM-2024-001", "CLM-2024-002"]), "digits") is None

    def test_prefixed_id_matches_a_bare_number(self):
        left = pl.DataFrame({"POL_NUM": list(range(100_000, 100_300))})
        right = pl.DataFrame({"POLICY_NO": [f"ON-{i:07d}" for i in range(100_000, 100_300, 3)]})
        best = detect_join_keys(left, right).best
        assert (best.left_column, best.right_column, best.key_form) == ("POL_NUM", "POLICY_NO", "digits")
        assert best.match_rate == 1.0

    def test_two_lettered_schemes_never_strip_to_match(self):
        # 'CL00012' and 'PN000012' are different things that share a number.
        left = pl.DataFrame({"POLICY_NO": [f"PN{i:06d}" for i in range(300)]})
        right = pl.DataFrame({"CLAIM_ID": [f"CL{i:05d}" for i in range(100)]})
        assert detect_join_keys(left, right).best is None
