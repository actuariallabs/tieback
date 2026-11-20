# tieback

**Reconcile insurance extracts. Join any number of files, correct the mistyped keys the data can vouch for, find everything else that is wrong, and prove the result ties back to the totals.**

A pricing or reserving project never starts with one clean file. It starts with five, from systems that were never designed to meet:

| Extract | Grain | The policy is called |
|---|---|---|
| Policy admin | one row per policy **term** | `ON-0482113` |
| Vehicle schedule | one row per vehicle per term, with the exposure | `482113` |
| Claims | one row per claim per **coverage**, a loss date, no term | `P0482113` |
| Finance premium | one row per policy term | `0482113/1` |
| Driver convictions | one row per conviction, per **policy**, dated | `0482113` |

And the files are wrong in the ways real files are wrong: a policy number typed `P0428113`, a vehicle exported twice, a region spelt `Toronto`, `TORONTO` and `Otawa`, a claim reported before it happened, paid above incurred, exposure longer than the term.

Join them naively and every one of those moves a rate without ever showing up as an error. `tieback` joins them, says what it did and why, and lists what is wrong with the numbers it affects.

```
$ pip install tieback
$ tieback demo sample          # five extracts with the problems real ones have
$ tieback sample/*.csv -o book.parquet --plan plan.json --findings findings.json
```

```
base: vehicles.csv (22,067 rows), exposure EARNED_CAR_YRS

claims.csv  [rollup]  on POLICY_REF = POL_NUM + VEHICLE = UNIT_NO, LOSS_DATE within [TERM_START, TERM_EXP)
    POLICY_REF ↔ POL_NUM: 'P0123423' and '123423' are the same policy once the prefix and leading zeros are dropped.
    claims has no term, only dates: LOSS_DATE between TERM_START and TERM_EXP places each row in one term.
    7 mistyped key values corrected first (9 rows). Chance check: a key that is missing, not mistyped,
    finds a convincing near match 0.5% of the time on this book, so about 0.2 of them could be wrong.
    CLAIM_NO: 2,144 in file = 2,074 attached + 35 no matching key + 35 outside every term + 0 ambiguous  ok
    INCURRED: 31,394,525 in file = 30,315,213 in table + 1,079,312 unattached  ok

premium.csv  [per group]  on GL_POLICY_TERM (part 1) = POL_NUM + BOOKED_FROM = TERM_START
    Each row belongs to a group of 1.49 vehicles rows on average (the rows sharing POL_NUM and TERM_START).
    Split across each group's rows by EARNED_CAR_YRS, so the total still ties: WRITTEN_PREMIUM.
    WRITTEN_PREMIUM: 29,034,341 in file = 29,013,424 in table + 20,917 unattached  ok

exposure: 21,125 in, 21,125 out  ok

findings: 4 errors, 9 to review, 10 fixed
  ERROR   claims.csv has 4 rows that repeat another row exactly.
          affects INCURRED 52,299, PAID 33,787
  ERROR   3 vehicles.csv rows overlap an earlier period of the same POL_NUM and UNIT_NO: exposure counted twice.
  ERROR   2 vehicles.csv rows carry more EARNED_CAR_YRS than their period is long.
  REVIEW  policies.csv.REGION spells 2 values two ways: 'TORONTO' (20) looks like 'Toronto' (3,182); 'Otawa' (5) looks like 'Ottawa' (1,711).
  REVIEW  6 claims.csv rows have PAID above INCURRED.
  REVIEW  3 claims.csv rows were reported before the loss happened (REPORTED_DATE < LOSS_DATE).
  REVIEW  claims.csv.POLICY_REF: 8 values match nothing but sit one character from a real key whose other columns
          and dates fit, and only that one. Not applied: a policy missing from the extract can sit one digit from a real one by chance.
  FIXED   claims.csv.POLICY_REF: 'P9016838' read as 'P0916838' (swapped neighbours; its other keys and dates match only this one).
```

That is real output on the sample book (trimmed). Five files, 22,000 exposure rows, about two seconds.

## What it does

**Joins any number of files.** It starts from the file that carries exposure and attaches every other one, in the order the data allows:

- **Keys from values, not names.** `POLICY_NO` and `POL_NUM` join because their values overlap. `ON-0482113`, `482113` and `P0482113` are recognised as the same policy.
- **Two numbers in one id.** `0482113/1` is split into policy and term, and each part is matched.
- **Refines until each row has one partner.** Renewals repeat the policy number, so the key is narrowed by another column, then by a date window: a loss belongs to the term it falls inside.
- **Three ways to attach.** Attributes look up one row each. Events (claims) roll up, counted once per claim id, amounts summed. Events at a coarser grain than the exposure (convictions per policy term, premium per term) attach to the whole group: counts as a per-group attribute, money split across the group by exposure so it still adds up to the file.

**Corrects mistyped keys, and tells you how far to trust it.** For each key that matches nothing it looks for real keys one keystroke away, of three kinds: case and spacing, swapped neighbours, one changed character.

- A candidate counts only if the rest of the row vouches for it: with the corrected number, the claim's vehicle and loss date fall in exactly one term. If two candidates fit, nothing is changed.
- Case, spacing and swapped neighbours are applied by default. A single changed character is listed as a likely correction and applied only with `--repair all`, because a policy simply missing from the extract often sits one digit from a real one.
- **The chance check.** `tieback` gives real rows random keys that exist nowhere and counts how often they would still be "corrected". That rate, on your own book, becomes the expected number of wrong corrections, printed next to the corrections. If it is too high, none are applied.
- A key that appears in another of your files is a real policy missing from this one, not a typo, and is left alone.

**Finds what else is wrong**, with what it does to the numbers:

| Check | Why it matters |
|---|---|
| Rows exported twice | Double exposure or double losses |
| Overlapping periods for the same unit | Exposure counted twice |
| Exposure longer than its period | Inflated exposure |
| Periods that end before they start | Broken terms |
| Negative amounts | Sign errors, or recoveries to treat on purpose |
| Paid above incurred | Stale reserves or mis-keyed amounts |
| Reported before the loss date | Swapped dates that move a claim to the wrong term |
| One category spelt several ways | Each spelling becomes its own rating level |
| Fields that disagree between files | Which system is the source of truth |
| Events with no exposure to sit on | Losses outside the book |
| Totals that do not tie to finance | `--control EARNED_CAR_YRS=21130` |

**Accounts for everything.** Base rows and exposure are unchanged. Every claim and every dollar is in the table or listed as unattached, with its reason.

**Saves its decisions.** The plan is JSON: keys, windows and every correction. Replay it on next quarter's extracts with `tieback --replay plan.json` and it lists every row that did not make it in.

## In Python

```python
import tieback

tables = tieback.read(["policies.csv", "vehicles.csv", "claims.csv", "premium.csv", "convictions.csv"])
book, plan = tieback.assemble(tables, controls={"EARNED_CAR_YRS": 21_130})
print(tieback.report(plan))

for f in plan.findings:              # dicts: check, severity, table, column, count, impact, examples, action
    if f["severity"] == "error":
        print(f["message"])

tieback.save_plan(plan, "plan.json")
result = tieback.replay(next_quarter_tables, tieback.load_plan("plan.json"))
```

Options: `repair=False` changes no key, `repair="all"` also applies single-character corrections, `exposure_hints={"vehicles.csv": "EARNED_CAR_YRS"}` names the exposure column if it is not detected.

## What it does not do

- It does not guess. Ambiguous keys, ambiguous rows and tables it cannot join are listed with the reason, never forced.
- It does not change your rating variables. Misspelt categories are reported with the mapping to use; applying it is your call.
- It does not decide what to do with orphan claims. Whether losses with no exposure are kept or dropped is an actuarial judgment.
- It does not send anything anywhere. No network, no telemetry. Polars and NumPy on your machine.

## How we test it

The demo book is generated with every problem above planted, and the tests check `tieback` against what was planted, not against its own report. Over four random books with about 50 planted typos: every typo in the exposure file and 26 of 32 swapped digits in the claims file were corrected, and 2 of 38 applied corrections were wrong. Both were claims on missing policies that sat, by chance, one swap from a real one whose vehicle and dates fit. The chance check estimated about one per book on those books.

On a book ten times denser (60,000 policies), coincidences are common (12.5%), and `tieback` declined to correct and listed the candidates instead. That is the intended behaviour: on dense numbering, keys and dates alone cannot vouch for a correction.

## Status

Version 0.2, alpha. The API may change before 1.0. CI runs on Python 3.10 to 3.13. About 10 seconds for five files and 380,000 rows on a laptop. If it gets a join wrong on your data, an issue describing the column shapes (not the data) is the most useful thing you can send.

## Who made this

[Actuarial Labs](https://actuariallabs.com) is an actuarial science research lab. `tieback` is the reconciliation step from Pia, our Chief AI Actuary, released on its own because every pricing, reserving and data team needs it.

Apache 2.0.
