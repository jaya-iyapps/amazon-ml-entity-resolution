"""Dataset profiling, ground-truth inspection, and chunked loading utilities.

Owned by: Person 1 (Data / Preprocessing)
"""

import collections
import os
from typing import Any, Dict, Iterator, List, Optional, Union

# Attempt pandas import; gracefully handle when pandas is not installed yet
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


def load_tsv(
    filepath: str,
    chunksize: Optional[int] = None,
    use_cols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> Union[Any, Iterator[Any]]:
    """Load a tab-separated file with optional chunking for large datasets.

    Supports chunked iteration to avoid out-of-memory errors on multi-million row datasets.

    Args:
        filepath: Path to the TSV file.
        chunksize: Optional number of rows per chunk. If specified and pandas is available,
            returns an iterator of DataFrames.
        use_cols: Optional list of columns to load.
        nrows: Optional maximum number of rows to read.

    Returns:
        DataFrame or iterator of DataFrames if pandas is available; otherwise parses line by line.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ImportError: If pandas is requested for DataFrame operations but unavailable.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TSV file not found: {filepath}")

    if PANDAS_AVAILABLE:
        return pd.read_csv(
            filepath,
            sep="\t",
            chunksize=chunksize,
            usecols=use_cols,
            nrows=nrows,
            dtype=str,  # Keep all identifiers as strings to prevent loss of leading zeros
            keep_default_na=False,  # Treat empty fields as empty strings, not NaN
        )
    else:
        # Fallback standard library generator when pandas is not yet installed
        return _stream_tsv_stdlib(filepath, nrows=nrows)


def _stream_tsv_stdlib(filepath: str, nrows: Optional[int] = None) -> Iterator[Dict[str, str]]:
    """Fallback streaming TSV parser using Python standard library."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        count = 0
        for line in f:
            if nrows is not None and count >= nrows:
                break
            parts = line.rstrip("\r\n").split("\t")
            row = {
                headers[i]: parts[i] if i < len(parts) else ""
                for i in range(len(headers))
            }
            yield row
            count += 1


def profile_dataframe(df: Any) -> Dict[str, Any]:
    """Generate high-level profile summary of a dataset DataFrame or record collection.

    Args:
        df: Pandas DataFrame or list of dictionary records.

    Returns:
        Dictionary containing record count, column names, missing count per column.
    """
    if PANDAS_AVAILABLE and isinstance(df, pd.DataFrame):
        summary = {
            "num_rows": len(df),
            "columns": list(df.columns),
            "missing_counts": {col: int((df[col] == "").sum() + df[col].isna().sum()) for col in df.columns},
        }
        if "country" in df.columns:
            summary["country_distribution"] = df["country"].value_counts().to_dict()
        return summary

    # Generic list of dicts handling
    records = list(df)
    if not records:
        return {"num_rows": 0, "columns": [], "missing_counts": {}}

    cols = list(records[0].keys())
    missing_counts = {c: 0 for c in cols}
    country_counts = collections.defaultdict(int)

    for r in records:
        for c in cols:
            val = r.get(c, "")
            if not val or str(val).strip() == "":
                missing_counts[c] += 1
        if "country" in r:
            country_counts[r["country"]] += 1

    summary = {
        "num_rows": len(records),
        "columns": cols,
        "missing_counts": missing_counts,
    }
    if country_counts:
        summary["country_distribution"] = dict(country_counts)
    return summary


def analyze_ground_truth(gt_path: str) -> Dict[str, Any]:
    """Analyze the train ground truth file to understand entity match distributions.

    Computes:
    - Total Source 1 reference entities
    - Number and percentage of singletons (entities with 0 matches)
    - Entities with at least 1 match
    - Total links and average matches per linked entity

    Args:
        gt_path: Path to `train_ground_truth.tsv`.

    Returns:
        Dictionary of ground truth statistics.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    total_s1 = 0
    singletons = 0
    total_links = 0
    match_counts = collections.Counter()

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\r\n").split("\t")
            matches_str = parts[1] if len(parts) > 1 else ""
            matches = [m.strip() for m in matches_str.split(",") if m.strip()]
            num_m = len(matches)
            if num_m == 0:
                singletons += 1
            else:
                total_links += num_m
            match_counts[num_m] += 1

    matched_s1 = total_s1 - singletons
    return {
        "total_source1_entities": total_s1,
        "singletons_count": singletons,
        "singletons_pct": (singletons / total_s1 * 100.0) if total_s1 > 0 else 0.0,
        "entities_with_matches": matched_s1,
        "entities_with_matches_pct": (matched_s1 / total_s1 * 100.0) if total_s1 > 0 else 0.0,
        "total_links": total_links,
        "avg_matches_per_linked_entity": (total_links / matched_s1) if matched_s1 > 0 else 0.0,
        "match_distribution": dict(match_counts),
    }
