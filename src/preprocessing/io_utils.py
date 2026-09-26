"""
Memory-efficient TSV I/O for the Amazon Entity Resolution dataset.

Design contract:
- All reading uses sep='\\t' explicitly — never CSV, never auto-detection.
- entity_id columns are always dtype=str to prevent numeric reinterpretation.
- matched_entity_ids is kept as raw string; empty string means zero matches.
- For large files use read_tsv_chunks(); read_tsv_full() is for small files only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


# Column dtype overrides that must always be applied to source files.
_SOURCE_DTYPES: dict[str, str] = {
    "entity_id": "str",
}

# Columns in the ground-truth file.
_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

_GT_DTYPES: dict[str, str] = {
    "source1_entity_id": "str",
    "matched_entity_ids": "str",
}


def read_tsv_chunks(
    path: str | Path,
    chunksize: int = 50_000,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Yield chunks of a TSV file as DataFrames.

    Parameters
    ----------
    path:
        Path to the TSV file.
    chunksize:
        Number of rows per chunk. Default 50,000 — safe for 400+ MB files
        at typical column widths (~200 chars/row -> ~10 MB/chunk).
    usecols:
        If given, only load these columns (reduces memory further).
    dtype:
        Extra dtype overrides. entity_id is always forced to str.

    Yields
    ------
    pd.DataFrame chunks; each chunk has a fresh integer index reset to 0..n.

    Notes
    -----
    - NaN is NOT substituted for empty strings — keep_default_na=False ensures
      empty matched_entity_ids stays "" not NaN.
    - na_values=[] disables pandas' built-in NaN inference so entity IDs like
      "NA" are not silently converted to float NaN.
    """
    path = Path(path)
    combined_dtype: dict[str, str] = {**_SOURCE_DTYPES, **(dtype or {})}

    reader = pd.read_csv(
        path,
        sep="\t",
        chunksize=chunksize,
        dtype=combined_dtype,
        usecols=usecols,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        engine="c",
    )
    for chunk in reader:
        yield chunk.reset_index(drop=True)


def read_tsv_full(
    path: str | Path,
    usecols: list[str] | None = None,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load an entire TSV file into a single DataFrame.

    Use ONLY for small files (< a few hundred MB). For the 400–500 MB source
    files use read_tsv_chunks() instead.

    Parameters
    ----------
    path:
        Path to the TSV file.
    usecols:
        If given, only load these columns.
    dtype:
        Extra dtype overrides.

    Returns
    -------
    pd.DataFrame with index reset to 0..n-1.
    """
    path = Path(path)
    combined_dtype: dict[str, str] = {**_SOURCE_DTYPES, **(dtype or {})}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=combined_dtype,
        usecols=usecols,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        engine="c",
    )
    return df.reset_index(drop=True)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv into a DataFrame.

    Columns returned:
    - source1_entity_id : str — always an S1- prefixed ID
    - matched_entity_ids : str — comma-separated S2-/S3- IDs, or '' if none
    - match_list : list[str] — parsed list of matched IDs (empty list if none)
    - match_count : int — number of matched entities (0 if none)

    Empty matched_entity_ids (no-match rows) are correctly represented as
    an empty list and count 0 — they are NOT dropped or converted to NaN.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=_GT_DTYPES,
        usecols=_GT_COLS,
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        engine="c",
    )
    df = df.reset_index(drop=True)

    # Parse comma-separated matched IDs into a Python list
    def _parse_ids(raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

    df["match_list"] = df["matched_entity_ids"].apply(_parse_ids)
    df["match_count"] = df["match_list"].apply(len)

    return df


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to a TSV file.

    Uses tab separator, UTF-8 encoding, no BOM, no index column.
    Creates parent directories if they do not exist.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
