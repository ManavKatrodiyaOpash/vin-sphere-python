"""
make_grouping.py
----------------
Applies Tier 1 + Tier 2 make consolidations to a VIN dataset.

Usage
-----
    import pandas as pd
    from make_grouping import normalize_make, apply_grouping, grouping_report

    df["make_grouped"] = apply_grouping(df["make"])
    grouping_report(df["make"], df["make_grouped"])

The original column is never modified in-place; the caller decides
whether to overwrite or keep both columns.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical mapping  {MEMBER_MAKE -> CANONICAL_MAKE}
#
# Keys   = every make that should be renamed (uppercase, stripped).
# Values = the canonical name to use instead.
# Makes that are already canonical do NOT need an entry.
# ---------------------------------------------------------------------------

MAKE_GROUPS: dict[str, str] = {

    # ── Tier 1 · Definite (same brand / duplicates) ─────────────────────────

    # SHARMAX duplicate (data-entry typo)
    "SHARAMAX":              "SHARMAX",

    # JIDU renamed to JIYUE in 2023 — same Geely-Baidu JV
    "JIDU":                  "JIYUE",

    # Dongfeng family (7 → 1)
    "DONGFENG E":            "DONGFENG",
    "DONGFENG-AEOLUS":       "DONGFENG",
    "DONGFENG-XIAOKA":       "DONGFENG",
    "AEOLUS":                "DONGFENG",   # Dongfeng Fengshen
    "FORTHING":              "DONGFENG",   # Dongfeng Fengxing
    "VOYAH":                 "DONGFENG",   # Dongfeng premium EV arm

    # ── Tier 2 · High confidence (sub-brands / shared manufacturer) ──────────

    # Piaggio Group
    "VESPA":                 "PIAGGIO",
    "APRILIA":               "PIAGGIO",
    "MOTO GUZZI":            "PIAGGIO",

    # Pierer Mobility (KTM Group)
    "HUSQVARNA":             "KTM",
    "GAS GAS":               "KTM",

    # Polaris
    "SLINGSHOT":             "POLARIS",

    # Harley-Davidson  (LiveWire spun off 2022 but same WMI origin)
    "LIVEWIRE":              "HARLEY-DAVIDSON",

    # Mercedes-Benz  (Maybach is a trim line since 2014, identical WMIs)
    "MAYBACH":               "MERCEDES BENZ",

    # Toyota  (Scion used Toyota WMIs; discontinued 2016)
    "SCION":                 "TOYOTA",

    # Chery  (EXEED = premium, OMODA = global, JAECOO = off-road sub-brands)
    "EXEED":                 "CHERY",
    "OMODA":                 "CHERY",
    "JAECOO":                "CHERY",

    # Great Wall Motor
    "HAVAL":                 "GWM",
    "TANK":                  "GWM",

    # FAW Group
    "BESTUNE":               "FAW",
    "HONGQI":                "FAW",

    # GAC Group
    "TRUMPCHI":              "GAC",
    "HYCAN":                 "GAC",

    # Changan Automobile
    "DEEPAL":                "CHANGAN",
    "AVATR":                 "CHANGAN",

    # SAIC Group
    "MG":                    "SAIC",
    "MAXUS":                 "SAIC",
    "WULING":                "SAIC",

    # Geely Holding Group
    "VOLVO":                 "GEELY",
    "LOTUS":                 "GEELY",
    "LYNK & CO":             "GEELY",
    "ZEEKR":                 "GEELY",
    "POLESTAR":              "GEELY",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_make(make: str) -> str:
    """
    Return the canonical make name for a single value.

    Lookup is case-insensitive and strips leading/trailing whitespace,
    so messy data is handled gracefully.

    Examples
    --------
    >>> normalize_make("SHARAMAX")
    'SHARMAX'
    >>> normalize_make("haval")
    'GWM'
    >>> normalize_make("BMW")   # not in mapping → returned as-is (uppercased)
    'BMW'
    """
    if not isinstance(make, str):
        return make                         # propagate NaN / None unchanged
    key = make.strip().upper()
    return MAKE_GROUPS.get(key, key)        # fall back to uppercased original


def apply_grouping(series) -> "pd.Series":
    """
    Apply make normalization to a pandas Series (or any iterable).

    Returns a new Series; the input is never modified.

    Parameters
    ----------
    series : pd.Series | list | iterable
        Column containing raw make strings.

    Returns
    -------
    pd.Series with canonical make names.
    """
    try:
        import pandas as pd
        s = pd.Series(series) if not hasattr(series, "map") else series
        return s.map(normalize_make)
    except ImportError:
        return [normalize_make(v) for v in series]


def grouping_report(before, after) -> None:
    """
    Print a consolidation summary comparing before / after make columns.

    Shows
    -----
    - Total makes before / after
    - Each group that was merged (members → canonical, with row counts)
    - Makes that were not changed
    """
    try:
        import pandas as pd
        before = pd.Series(before)
        after  = pd.Series(after)
    except ImportError:
        pass

    before_counts = _count(before)
    after_counts  = _count(after)

    n_before = len(before_counts)
    n_after  = len(after_counts)

    print("=" * 60)
    print(f"  Make consolidation report")
    print(f"  {n_before} makes  →  {n_after} makes  (−{n_before - n_after})")
    print("=" * 60)

    # Build reverse map: canonical → [members that were remapped TO it]
    absorbed: dict[str, list[str]] = {}
    for member, canonical in MAKE_GROUPS.items():
        absorbed.setdefault(canonical, []).append(member)

    print("\nGroups merged:\n")
    for canonical, members in sorted(absorbed.items()):
        # Only report members that actually appeared in the data
        found = [m for m in members if m in before_counts]
        if not found:
            continue
        rows_absorbed = sum(before_counts.get(m, 0) for m in found)
        rows_canonical = before_counts.get(canonical, 0)
        total = rows_canonical + rows_absorbed
        member_str = ", ".join(
            f"{m} ({before_counts[m]})" for m in sorted(found)
        )
        print(f"  {canonical} ({total} rows total)")
        print(f"    ← {member_str}")

    unchanged = [
        m for m in after_counts
        if m not in {v for v in MAKE_GROUPS.values()}
        and m not in MAKE_GROUPS
    ]
    print(f"\nUnchanged makes: {len(unchanged)}")
    print("=" * 60)


def _count(series) -> dict:
    """Return {value: count} from a Series or list."""
    try:
        return series.value_counts().to_dict()
    except AttributeError:
        from collections import Counter
        return dict(Counter(series))


# ---------------------------------------------------------------------------
# Quick smoke-test  (python make_grouping.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = [
        "SHARAMAX", "SHARMAX",
        "JIDU", "JIYUE",
        "DONGFENG", "DONGFENG E", "DONGFENG-AEOLUS", "DONGFENG-XIAOKA",
        "AEOLUS", "FORTHING", "VOYAH",
        "VESPA", "APRILIA", "MOTO GUZZI", "PIAGGIO",
        "HUSQVARNA", "GAS GAS", "KTM",
        "SLINGSHOT", "POLARIS",
        "LIVEWIRE", "HARLEY-DAVIDSON",
        "MAYBACH", "MERCEDES BENZ",
        "SCION", "TOYOTA",
        "EXEED", "OMODA", "JAECOO", "CHERY",
        "HAVAL", "TANK", "GWM",
        "BESTUNE", "HONGQI", "FAW",
        "TRUMPCHI", "HYCAN", "GAC",
        "DEEPAL", "AVATR", "CHANGAN",
        "MG", "MAXUS", "WULING", "SAIC",
        "VOLVO", "LOTUS", "LYNK & CO", "ZEEKR", "POLESTAR", "GEELY",
        "BMW", "AUDI", "TESLA",           # unchanged makes
    ]

    result = apply_grouping(sample)

    print("Spot-check mappings:")
    checks = [
        ("SHARAMAX",    "SHARMAX"),
        ("JIDU",        "JIYUE"),
        ("VOYAH",       "DONGFENG"),
        ("FORTHING",    "DONGFENG"),
        ("VESPA",       "PIAGGIO"),
        ("GAS GAS",     "KTM"),
        ("SLINGSHOT",   "POLARIS"),
        ("LIVEWIRE",    "HARLEY-DAVIDSON"),
        ("MAYBACH",     "MERCEDES BENZ"),
        ("SCION",       "TOYOTA"),
        ("EXEED",       "CHERY"),
        ("HAVAL",       "GWM"),
        ("BESTUNE",     "FAW"),
        ("HYCAN",       "GAC"),
        ("DEEPAL",      "CHANGAN"),
        ("MG",          "SAIC"),
        ("POLESTAR",    "GEELY"),
        ("BMW",         "BMW"),            # unchanged
        ("TESLA",       "TESLA"),          # unchanged
        ("haval",       "GWM"),            # lowercase tolerance check
    ]

    all_pass = True
    for raw, expected in checks:
        got = normalize_make(raw)
        status = "✓" if got == expected else "✗"
        if got != expected:
            all_pass = False
        print(f"  {status}  {raw!r:25s} → {got!r}  (expected {expected!r})")

    print()
    grouping_report(sample, result)

    if all_pass:
        print("\nAll spot-checks passed.")
    else:
        print("\nSome checks FAILED — review mapping.")