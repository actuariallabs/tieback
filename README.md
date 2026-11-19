# tieback

**Join insurance extracts that were never meant to be joined, and prove the result ties back to the totals.**

A pricing or reserving project rarely starts with one clean file. It starts with three, from systems that were never designed to meet:

| Extract | Grain | The policy is called |
|---|---|---|
| Policy admin | one row per policy **term** | `ON-0482113` |
| Vehicle schedule | one row per vehicle per term, with the exposure | `482113` |
| Claims | one row per claim per **coverage**, a loss date, no term | `P0482113` |

Join them naively and you get the classic errors. Renewals repeat the policy number, so every claim lands on every term. A claim with three coverages counts as three claims. Exposure gets summed over a fanned-out table. Each error moves a rate, and none of them shows up as an error.

`tieback` finds the keys from the values (not the column names), narrows them with more columns and date windows until every row has exactly one partner, and then accounts for every row, every claim and every dollar.

```
$ pip install tieback
$ tieback demo sample            # writes a 3-file Ontario auto book with planted problems
$ tieback sample/policies.csv sample/vehicles.csv sample/claims.csv -o book.parquet --plan plan.json
```

```
base: vehicles.csv (22,064 rows), exposure EARNED_CAR_YRS

policies.csv  [lookup]  on POLICY_NO = POL_NUM + TERM_EFF = TERM_START
    POLICY_NO ↔ POL_NUM: 'ON-0645674' and '645674' are the same policy once the prefix and leading zeros are dropped.
    On its own it matches 79.0% of vehicles rows to several policies rows (renewals repeat the number).
    TERM_EFF ↔ TERM_START narrows it.
    rows: 22,064 before, 22,064 after  ok

claims.csv  [rollup]  on POLICY_REF = POL_NUM + VEHICLE = UNIT_NO, LOSS_DATE within [TERM_START, TERM_EXP)
    claims has no term, only dates: LOSS_DATE between TERM_START and TERM_EXP places each row in one term.
    REPORTED_DATE instead would place 95.5% of rows, against 97.3% for LOSS_DATE.
    CLAIM_NO repeats across rows (2,930 rows for 2,144 claims): each is counted once.
    CLAIM_NO: 2,144 in file = 2,079 attached + 30 no matching key + 35 outside every term + 0 ambiguous  ok
    INCURRED: 31,357,852 in file = 30,321,487 in table + 1,036,365 unattached  ok

exposure: 21,121 in, 21,121 out  ok
rows: 22,064 in, 22,064 out  ok
```

That is real output on the sample book (trimmed), in about a second.

## In Python

```python
import tieback

tables = tieback.read(["policies.csv", "vehicles.csv", "claims.csv"])
book, plan = tieback.assemble(tables)      # a Polars DataFrame and the plan
print(tieback.report(plan))

# Does earned exposure agree with finance?
print(tieback.check_total(book, "EARNED_CAR_YRS", expected=21_130, tolerance=0.001))

# Next quarter: same decisions, no re-detection, and a list of every row that missed.
tieback.save_plan(plan, "plan.json")
result = tieback.replay(next_quarter_tables, tieback.load_plan("plan.json"))
```

Two tables only? `tieback.detect_join_keys(left, right)` ranks candidate keys with match rate, orphans in both directions and cardinality, and `tieback.execute_join` aggregates a one-to-many join by default, so exposure is never double-counted.

## What it does

- **Keys from values.** `POLICY_NO` and `POL_NUM` join because their values overlap, not because their names look alike. Ids that differ only by a prefix or leading zeros are matched and shown with an example pair.
- **Refines until unique.** A key that matches several partners is narrowed by another column, then by a date window (a loss belongs to the term it falls inside). Rows that stay ambiguous are reported and never joined.
- **Lookups, then roll-ups.** Attributes attach one-to-one onto the exposure table. Events (claims) are counted once per claim id and their amounts summed.
- **Control totals at every step.** Base rows and exposure are unchanged. Every claim and every dollar in the file is either in the table or listed as unattached, with its reason.
- **Plans are JSON.** Save the decisions, replay them on next quarter's extracts, and hand the file to whoever audits the work.

## What it does not do

- It does not guess when it cannot tell. A table with no resolvable key is left out and named in the report, with the reason.
- It does not send anything anywhere. No network, no telemetry. It is Polars and NumPy on your machine.
- It does not decide what to do with orphan claims. It lists them. Whether claims with no exposure are kept or dropped is an actuarial judgment.

## Status

Version 0.1, alpha. The API may change before 1.0. CI runs the tests on Python 3.10 to 3.13. Key detection has also been checked on freMTPL2 (678k policies). If it gets a join wrong on your data, an issue with the column shapes (not the data) is the most useful thing you can send.

## Who made this

[Actuarial Labs](https://actuariallabs.com) is an actuarial science research lab. `tieback` is the reconciliation step from Pia, our AI actuary, released on its own because every pricing, reserving and data team needs it. Pia runs it every month on the live extracts and tells the actuary when something stops tying.

Apache 2.0.
