"""Multi-dataset reconciliation tests.

The premise: detection must work on column names that share nothing
(`POLICY_NO` vs `POL_NUM`), because that is what real carrier extracts look
like. So the fixtures here deliberately use mismatched names, and the
name-similarity path is only exercised where it is explicitly the subject.
"""

import numpy as np
import polars as pl
import pytest

from tieback.keys import (
    JoinSpec,
    detect_join_keys,
    execute_join,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def policy_df() -> pl.DataFrame:
    rng = _rng(4)
    n = 3000
    return pl.DataFrame(
        {
            "POLICY_NO": [f"PN{i:06d}" for i in range(n)],
            "DRV_AGE": rng.integers(18, 80, n),
            "EARNED_CAR_YRS": rng.uniform(0.3, 1.0, n),
        }
    )


@pytest.fixture
def claims_df(policy_df) -> pl.DataFrame:
    """300 claims against 3000 policies, plus 12 orphans."""
    rng = _rng(5)
    pols = rng.choice(3000, 300, replace=True)
    return pl.DataFrame(
        {
            "CLAIM_ID": [f"CL{i:05d}" for i in range(312)],
            "POL_NUM": [f"PN{i:06d}" for i in pols]
            + [f"PN9{i:05d}" for i in range(12)],
            "PAID_AMT": rng.gamma(2.0, 1500.0, 312),
        }
    )


class TestKeyDetection:
    def test_finds_key_despite_unrelated_names(self, policy_df, claims_df):
        best = detect_join_keys(policy_df, claims_df).best
        assert best is not None
        assert best.left_column == "POLICY_NO"
        assert best.right_column == "POL_NUM"

    def test_is_confident(self, policy_df, claims_df):
        assert detect_join_keys(policy_df, claims_df).is_confident

    def test_infers_one_to_many(self, policy_df, claims_df):
        assert detect_join_keys(policy_df, claims_df).best.cardinality == "one_to_many"

    def test_flags_aggregation_requirement(self, policy_df, claims_df):
        assert detect_join_keys(policy_df, claims_df).best.requires_aggregation

    def test_counts_orphans(self, policy_df, claims_df):
        assert detect_join_keys(policy_df, claims_df).best.orphan_right_rows == 12

    def test_claim_id_is_not_a_join_key(self, policy_df, claims_df):
        """CLAIM_ID is unique on the right but shares no values with the
        left — it must not surface as a candidate."""
        pairs = {
            (c.left_column, c.right_column)
            for c in detect_join_keys(policy_df, claims_df).candidates
        }
        assert not any(r == "CLAIM_ID" for _, r in pairs)

    def test_low_cardinality_column_is_never_a_key(self, policy_df, claims_df):
        """A 3-level rating factor must not be proposed — joining on it
        would explode the row count."""
        left = policy_df.with_columns(pl.lit("A").alias("REGION"))
        right = claims_df.with_columns(pl.lit("A").alias("REGION"))
        pairs = {
            (c.left_column, c.right_column)
            for c in detect_join_keys(left, right).candidates
        }
        assert ("REGION", "REGION") not in pairs

    def test_float_columns_are_not_keys(self, policy_df, claims_df):
        """Floats are measurements. Two exposure columns overlapping on 1.0
        is coincidence, not a relationship."""
        pairs = {
            (c.left_column, c.right_column)
            for c in detect_join_keys(policy_df, claims_df).candidates
        }
        assert ("EARNED_CAR_YRS", "PAID_AMT") not in pairs

    def test_reasons_are_populated(self, policy_df, claims_df):
        reasons = " ".join(detect_join_keys(policy_df, claims_df).best.reasons).lower()
        assert "match" in reasons
        assert "1:n" in reasons


class TestTypeCoercion:
    def test_integer_key_matches_string_key(self):
        """A common export mismatch: integer id on one side, the same id as
        a string on the other."""
        rng = _rng(8)
        left = pl.DataFrame(
            {"pol_id": list(range(1, 2001)), "expo": rng.uniform(0.3, 1.0, 2000)}
        )
        right = pl.DataFrame(
            {
                "policy_number": [str(i) for i in rng.choice(range(1, 2001), 300)],
                "amt": rng.gamma(2.0, 1000.0, 300),
            }
        )
        best = detect_join_keys(left, right).best
        assert best is not None
        assert best.match_rate == pytest.approx(1.0)

    def test_whitespace_is_stripped(self):
        left = pl.DataFrame({"k": [f"K{i:04d}" for i in range(200)], "v": range(200)})
        right = pl.DataFrame(
            {"k2": [f" K{i:04d} " for i in range(150)], "w": range(150)}
        )
        best = detect_join_keys(left, right).best
        assert best is not None
        assert best.match_rate == pytest.approx(1.0)


class TestJoinExecution:
    def test_aggregation_preserves_left_row_count(self, policy_df, claims_df):
        """The core guarantee: a 1:N join must not fan the left side out,
        or exposure is double-counted."""
        spec = JoinSpec("POLICY_NO", "POL_NUM", sum_columns=["PAID_AMT"])
        out, diags = execute_join(policy_df, claims_df, spec)
        assert out.height == policy_df.height
        assert diags.result_rows == diags.left_rows

    def test_exposure_total_is_unchanged(self, policy_df, claims_df):
        """The actual thing double-counting breaks."""
        before = policy_df["EARNED_CAR_YRS"].sum()
        out, _ = execute_join(
            policy_df, claims_df, JoinSpec("POLICY_NO", "POL_NUM")
        )
        assert out["EARNED_CAR_YRS"].sum() == pytest.approx(before)

    def test_emits_count_and_sum(self, policy_df, claims_df):
        spec = JoinSpec("POLICY_NO", "POL_NUM", sum_columns=["PAID_AMT"])
        out, _ = execute_join(policy_df, claims_df, spec)
        assert "claim_count" in out.columns
        assert "PAID_AMT_sum" in out.columns

    def test_count_totals_matched_rows(self, policy_df, claims_df):
        spec = JoinSpec("POLICY_NO", "POL_NUM", sum_columns=["PAID_AMT"])
        out, diags = execute_join(policy_df, claims_df, spec)
        assert out["claim_count"].sum() == diags.matched_right_rows

    def test_fan_out_when_aggregation_disabled(self, policy_df, claims_df):
        spec = JoinSpec("POLICY_NO", "POL_NUM", aggregate=False)
        out, _ = execute_join(policy_df, claims_df, spec)
        assert out.height > policy_df.height

    def test_unmatched_left_rows_survive_a_left_join(self, policy_df, claims_df):
        """Every policy survives, and a policy with no claims records ZERO
        claims rather than null.

        Null was the original behaviour and it was a silent correctness bug:
        claim-free policies were dropped from the fit, so the model trained
        only on claimants — where frequency is 1 by construction — and
        reported a perfect Gini on a model that had learned nothing.
        """
        out, _ = execute_join(
            policy_df, claims_df, JoinSpec("POLICY_NO", "POL_NUM", how="left")
        )
        assert out.height == policy_df.height
        assert out["claim_count"].null_count() == 0
        # The unmatched policies are present and counted as zero-claim.
        assert (out["claim_count"] == 0).sum() > 0

    def test_inner_join_drops_unmatched_left(self, policy_df, claims_df):
        out, _ = execute_join(
            policy_df, claims_df, JoinSpec("POLICY_NO", "POL_NUM", how="inner")
        )
        assert out.height < policy_df.height

    def test_missing_column_raises(self, policy_df, claims_df):
        with pytest.raises(ValueError, match="left_column"):
            execute_join(policy_df, claims_df, JoinSpec("NOPE", "POL_NUM"))
        with pytest.raises(ValueError, match="right_column"):
            execute_join(policy_df, claims_df, JoinSpec("POLICY_NO", "NOPE"))


class TestDiagnostics:
    def test_orphan_claims_are_reported(self, policy_df, claims_df):
        _, d = execute_join(policy_df, claims_df, JoinSpec("POLICY_NO", "POL_NUM"))
        assert d.orphan_right_rows == 12
        assert d.is_healthy is False
        assert any("orphan" in w.lower() for w in d.warnings)

    def test_truncated_extract_warns(self):
        """Claims referencing policies outside the in-force snapshot — the
        silent-wrong-answer case this whole module exists to prevent."""
        rng = _rng(12)
        pol = pl.DataFrame(
            {"POLICY_NO": [f"PN{i:06d}" for i in range(1500)], "e": rng.uniform(0.3, 1.0, 1500)}
        )
        clm = pl.DataFrame(
            {
                "POL_NUM": [f"PN{i:06d}" for i in rng.choice(3000, 200)],
                "PAID": rng.gamma(2.0, 900.0, 200),
            }
        )
        best = detect_join_keys(pol, clm).best
        assert best is not None
        _, d = execute_join(pol, clm, JoinSpec(best.left_column, best.right_column))
        assert d.match_rate < 0.9
        assert d.is_healthy is False
        assert d.warnings

    def test_clean_one_to_one_is_healthy(self):
        left = pl.DataFrame({"k": [f"K{i:04d}" for i in range(500)], "a": range(500)})
        right = pl.DataFrame({"k2": [f"K{i:04d}" for i in range(500)], "b": range(500)})
        _, d = execute_join(left, right, JoinSpec("k", "k2"))
        assert d.match_rate == pytest.approx(1.0)
        assert d.is_healthy
        assert d.warnings == []


class TestSortedLargeExtracts:
    """Regressions from validating against real freMTPL2 (678k rows).

    Synthetic fixtures hid both of these: small frames skip sampling, and
    hand-built DataFrames get clean integer dtypes.
    """

    def test_sorted_key_beyond_sample_cap_is_still_found(self):
        """Real extracts are sorted by policy id. Sampling BOTH sides took
        disjoint slices of the key space — a true 100%-match join measured
        7.1% and fell below threshold. The left key set must stay complete."""
        n = 60_000
        left = pl.DataFrame(
            {"IDpol": list(range(1, n + 1)), "Exposure": [0.5] * n}
        )
        right = pl.DataFrame(
            {"IDpol": list(range(1, n + 1, 3)), "Amount": [100.0] * len(range(1, n + 1, 3))}
        )
        result = detect_join_keys(left, right, sample_rows=5_000)
        assert result.best is not None
        assert result.best.match_rate == pytest.approx(1.0)

    def test_integer_valued_float_id_is_a_valid_key(self):
        """CSV inference types an integer id column as Float64 when it has
        nulls — freMTPL2's IDpol arrives that way. Rejecting all floats
        rejected the real key."""
        left = pl.DataFrame(
            {"IDpol": [float(i) for i in range(1, 501)], "e": [0.5] * 500}
        )
        right = pl.DataFrame(
            {"IDpol": [float(i) for i in range(1, 251)], "amt": [10.0] * 250}
        )
        best = detect_join_keys(left, right).best
        assert best is not None
        assert best.match_rate == pytest.approx(1.0)

    def test_float_id_matches_string_id_across_sides(self):
        """1234.0 must join to "1234" — normalisation has to agree between
        detection and execution."""
        left = pl.DataFrame({"pid": [float(i) for i in range(1, 401)], "e": [1.0] * 400})
        right = pl.DataFrame({"pid2": [str(i) for i in range(1, 201)], "a": [5.0] * 200})
        best = detect_join_keys(left, right).best
        assert best is not None
        out, d = execute_join(left, right, JoinSpec(best.left_column, best.right_column))
        assert d.match_rate == pytest.approx(1.0)
        assert out.height == 400

    def test_genuine_float_measurement_is_still_rejected(self):
        """The float guard must survive the integer-valued-float exception."""
        left = pl.DataFrame({"expo": [i / 7.0 for i in range(1, 301)]})
        right = pl.DataFrame({"expo2": [i / 7.0 for i in range(1, 301)]})
        pairs = {
            (c.left_column, c.right_column) for c in detect_join_keys(left, right).candidates
        }
        assert ("expo", "expo2") not in pairs


class TestEdgeCases:
    def test_empty_frames(self):
        assert detect_join_keys(pl.DataFrame(), pl.DataFrame()).candidates == []

    def test_no_shared_values(self):
        left = pl.DataFrame({"k": [f"A{i}" for i in range(200)]})
        right = pl.DataFrame({"k2": [f"B{i}" for i in range(200)]})
        assert detect_join_keys(left, right).best is None

    def test_nulls_in_key_do_not_crash(self):
        left = pl.DataFrame({"k": [f"K{i:04d}" for i in range(300)], "a": range(300)})
        right = pl.DataFrame(
            {
                "k2": [f"K{i:04d}" if i % 7 else None for i in range(300)],
                "b": range(300),
            }
        )
        best = detect_join_keys(left, right).best
        assert best is not None

    def test_one_to_one_cardinality(self):
        left = pl.DataFrame({"k": [f"K{i:04d}" for i in range(400)], "a": range(400)})
        right = pl.DataFrame({"k2": [f"K{i:04d}" for i in range(400)], "b": range(400)})
        assert detect_join_keys(left, right).best.cardinality == "one_to_one"

    def test_sampling_still_finds_the_key(self):
        rng = _rng(21)
        n = 20_000
        left = pl.DataFrame(
            {"k": [f"K{i:06d}" for i in range(n)], "a": rng.uniform(0, 1, n)}
        )
        right = pl.DataFrame(
            {"k2": [f"K{i:06d}" for i in range(0, n, 2)], "b": range(n // 2)}
        )
        best = detect_join_keys(left, right, sample_rows=2_000).best
        assert best is not None
        assert (best.left_column, best.right_column) == ("k", "k2")
