"""The report, the saved plan and the command line, on the sample book."""

from __future__ import annotations

import polars as pl
import pytest

import tieback
from tieback.cli import main
from tieback.demo import build


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    out = tmp_path_factory.mktemp("book")
    build(out, n_policies=800, seed=5)
    return [out / n for n in ("policies.csv", "vehicles.csv", "claims.csv")]


@pytest.fixture(scope="module")
def assembled(files):
    return tieback.assemble(tieback.read(files))


def test_report_accounts_for_everything(assembled):
    _, plan = assembled
    text = tieback.report(plan)
    assert "base: vehicles.csv" in text
    assert "CHECK" not in text, text
    assert text.count(" ok") >= 4


def test_plan_round_trips_and_replays(tmp_path, files, assembled):
    book, plan = assembled
    path = tmp_path / "plan.json"
    tieback.save_plan(plan, path)
    saved = tieback.load_plan(path)
    assert saved["base"] == plan.base
    result = tieback.replay(tieback.read(files), saved)
    assert set(result["grain"]) == {f.name for f in files}
    missed_claims = sum(m["rows"].height for s in result["steps"] if s["kind"] == "rollup" for m in s["misses"])
    assert missed_claims > 0  # the sample book plants orphan and out-of-term claims


def test_check_total(assembled):
    book, plan = assembled
    total = float(book[plan.exposure].sum())
    assert tieback.check_total(book, plan.exposure, total * 1.0005).ties
    off = tieback.check_total(book, plan.exposure, total * 1.01)
    assert not off.ties and "does not tie" in str(off)
    with pytest.raises(ValueError):
        tieback.check_total(book, "nope", 1.0)


def test_cli_writes_table_and_plan(tmp_path, files, capsys):
    out, plan = tmp_path / "book.parquet", tmp_path / "plan.json"
    assert main([*map(str, files), "-o", str(out), "--plan", str(plan)]) == 0
    assert pl.read_parquet(out).height == pl.read_csv(files[1]).height
    assert main(["--replay", str(plan), *map(str, files)]) == 0
    assert "not used" in capsys.readouterr().out
