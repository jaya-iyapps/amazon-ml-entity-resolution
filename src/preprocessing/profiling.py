"""
Data profiling utilities for the Amazon Entity Resolution dataset.

Design:
- All functions operate on a single DataFrame chunk so they can be called
  incrementally without loading entire 400–500 MB files into memory.
- Profiles are plain dicts — easy to serialize, merge, and display.
- merge_profiles() combines chunk results into an aggregate profile.
- No global state; all functions are pure.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Profile = dict[str, Any]


# ---------------------------------------------------------------------------
# Source-file chunk profiler
# ---------------------------------------------------------------------------

def profile_source_chunk(chunk: pd.DataFrame) -> Profile:
    """
    Produce a partial profile for one chunk of a source TSV.

    Returns a dict with:
    - row_count              : int
    - missing_entity_id      : int
    - missing_business_name  : int
    - missing_address        : int
    - missing_country        : int
    - country_counts         : Counter {country_str: count}
    - name_length_sum        : int   (chars, for mean later)
    - name_length_min        : int
    - name_length_max        : int
    - address_length_sum     : int
    - address_length_min     : int
    - address_length_max     : int
    - entity_id_prefixes     : Counter {prefix: count}  e.g. {"S1-":10, "S2-":5}
    """
    p: Profile = {}

    p["row_count"] = len(chunk)

    # Missing value counts (empty string or NaN)
    def _missing(col: str) -> int:
        if col not in chunk.columns:
            return len(chunk)
        return int(chunk[col].apply(lambda v: v is None or (isinstance(v, float) and v != v) or str(v).strip() == "").sum())

    p["missing_entity_id"] = _missing("entity_id")
    p["missing_business_name"] = _missing("business_name")
    p["missing_address"] = _missing("business_address")
    p["missing_country"] = _missing("country")

    # Country distribution
    if "country" in chunk.columns:
        country_counts: Counter = Counter()
        for v in chunk["country"].dropna():
            country_counts[str(v).strip()] += 1
        p["country_counts"] = country_counts
    else:
        p["country_counts"] = Counter()

    # Business name lengths
    if "business_name" in chunk.columns:
        lengths = chunk["business_name"].apply(lambda v: len(str(v)) if v and str(v).strip() else 0)
        p["name_length_sum"] = int(lengths.sum())
        p["name_length_min"] = int(lengths.min()) if len(lengths) > 0 else 0
        p["name_length_max"] = int(lengths.max()) if len(lengths) > 0 else 0
    else:
        p["name_length_sum"] = 0
        p["name_length_min"] = 0
        p["name_length_max"] = 0

    # Address lengths
    if "business_address" in chunk.columns:
        lengths = chunk["business_address"].apply(lambda v: len(str(v)) if v and str(v).strip() else 0)
        p["address_length_sum"] = int(lengths.sum())
        p["address_length_min"] = int(lengths.min()) if len(lengths) > 0 else 0
        p["address_length_max"] = int(lengths.max()) if len(lengths) > 0 else 0
    else:
        p["address_length_sum"] = 0
        p["address_length_min"] = 0
        p["address_length_max"] = 0

    # Entity ID prefix distribution
    if "entity_id" in chunk.columns:
        prefix_counts: Counter = Counter()
        for eid in chunk["entity_id"].dropna():
            s = str(eid)
            if s.startswith("S1-"):
                prefix_counts["S1-"] += 1
            elif s.startswith("S2-"):
                prefix_counts["S2-"] += 1
            elif s.startswith("S3-"):
                prefix_counts["S3-"] += 1
            else:
                prefix_counts["other"] += 1
        p["entity_id_prefixes"] = prefix_counts
    else:
        p["entity_id_prefixes"] = Counter()

    return p


# ---------------------------------------------------------------------------
# Ground-truth chunk profiler
# ---------------------------------------------------------------------------

def profile_ground_truth_chunk(chunk: pd.DataFrame) -> Profile:
    """
    Produce a partial profile for one chunk of ground_truth TSV.

    The chunk may have the raw 'matched_entity_ids' column (str), or
    the parsed 'match_list' / 'match_count' columns added by load_ground_truth.

    Returns a dict with:
    - row_count              : int
    - singletons             : int   (rows with 0 matches)
    - has_matches            : int   (rows with >= 1 match)
    - total_s2_matches       : int   (across all rows)
    - total_s3_matches       : int   (across all rows)
    - match_count_distribution: Counter {n_matches: row_count}
    """
    p: Profile = {}
    p["row_count"] = len(chunk)

    # Resolve match lists
    if "match_list" in chunk.columns:
        match_lists = chunk["match_list"].tolist()
    elif "matched_entity_ids" in chunk.columns:
        def _parse(raw: Any) -> list[str]:
            s = str(raw).strip() if raw is not None else ""
            if not s:
                return []
            return [x.strip() for x in s.split(",") if x.strip()]
        match_lists = [_parse(v) for v in chunk["matched_entity_ids"]]
    else:
        match_lists = [[] for _ in range(len(chunk))]

    singletons = 0
    has_matches = 0
    s2_total = 0
    s3_total = 0
    match_dist: Counter = Counter()

    for ml in match_lists:
        n = len(ml)
        match_dist[n] += 1
        if n == 0:
            singletons += 1
        else:
            has_matches += 1
        for eid in ml:
            if eid.startswith("S2-"):
                s2_total += 1
            elif eid.startswith("S3-"):
                s3_total += 1

    p["singletons"] = singletons
    p["has_matches"] = has_matches
    p["total_s2_matches"] = s2_total
    p["total_s3_matches"] = s3_total
    p["match_count_distribution"] = match_dist

    return p


# ---------------------------------------------------------------------------
# Profile merger
# ---------------------------------------------------------------------------

def merge_profiles(profiles: list[Profile]) -> Profile:
    """
    Merge a list of chunk profiles (from the same profiler) into one aggregate.

    Supports both source profiles and ground-truth profiles by merging only
    keys that are present. Numeric keys are summed; Counter keys are summed;
    min/max keys take the correct extreme.
    """
    if not profiles:
        return {}

    merged: Profile = {}

    sum_keys = {
        "row_count", "missing_entity_id", "missing_business_name",
        "missing_address", "missing_country",
        "name_length_sum", "address_length_sum",
        "singletons", "has_matches", "total_s2_matches", "total_s3_matches",
    }
    min_keys = {"name_length_min", "address_length_min"}
    max_keys = {"name_length_max", "address_length_max"}
    counter_keys = {
        "country_counts", "entity_id_prefixes", "match_count_distribution"
    }

    all_keys: set[str] = set()
    for p in profiles:
        all_keys.update(p.keys())

    for key in all_keys:
        values = [p[key] for p in profiles if key in p]
        if not values:
            continue
        if key in sum_keys:
            merged[key] = sum(values)
        elif key in min_keys:
            merged[key] = min(values)
        elif key in max_keys:
            merged[key] = max(values)
        elif key in counter_keys:
            total: Counter = Counter()
            for v in values:
                total.update(v)
            merged[key] = total
        else:
            # Unknown key: keep last value (best-effort)
            merged[key] = values[-1]

    # Derived stats
    row_count = merged.get("row_count", 0)
    if row_count > 0:
        name_sum = merged.get("name_length_sum", 0)
        addr_sum = merged.get("address_length_sum", 0)
        merged["name_length_mean"] = round(name_sum / row_count, 2)
        merged["address_length_mean"] = round(addr_sum / row_count, 2)

    return merged


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_profile(profile: Profile, title: str = "Profile") -> None:
    """
    Print a human-readable summary of a merged profile to stdout.
    Safe for both source and ground-truth profiles.
    """
    print(f"\n=== {title} ===")

    if "row_count" in profile:
        print(f"  Rows:                {profile['row_count']:,}")

    for key in ("missing_entity_id", "missing_business_name",
                "missing_address", "missing_country"):
        if key in profile:
            col = key.replace("missing_", "")
            pct = 100 * profile[key] / max(profile.get("row_count", 1), 1)
            print(f"  Missing {col:<22}: {profile[key]:,}  ({pct:.2f}%)")

    if "entity_id_prefixes" in profile:
        print("  Entity ID prefixes:")
        for prefix, count in sorted(profile["entity_id_prefixes"].items()):
            print(f"    {prefix}: {count:,}")

    if "country_counts" in profile:
        print("  Country distribution (top 15):")
        for country, count in profile["country_counts"].most_common(15):
            display = country if country else "(empty)"
            print(f"    {display:<30}: {count:,}")

    for stat, label in [
        ("name_length_min", "Name length min"),
        ("name_length_max", "Name length max"),
        ("name_length_mean", "Name length mean"),
        ("address_length_min", "Address length min"),
        ("address_length_max", "Address length max"),
        ("address_length_mean", "Address length mean"),
    ]:
        if stat in profile:
            print(f"  {label:<28}: {profile[stat]}")

    if "singletons" in profile:
        print(f"  Singletons (0 matches):     {profile['singletons']:,}")
    if "has_matches" in profile:
        print(f"  Has matches:                {profile['has_matches']:,}")
    if "total_s2_matches" in profile:
        print(f"  Total S2 matches:           {profile['total_s2_matches']:,}")
    if "total_s3_matches" in profile:
        print(f"  Total S3 matches:           {profile['total_s3_matches']:,}")

    if "match_count_distribution" in profile:
        print("  Match count distribution (top 10):")
        dist = profile["match_count_distribution"]
        for n_matches, count in sorted(dist.items())[:10]:
            print(f"    {n_matches} match(es): {count:,} rows")

    print()
