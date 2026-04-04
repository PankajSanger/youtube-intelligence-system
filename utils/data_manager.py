from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/youtube_data_for_project.xlsx")

STANDARD_COLUMNS = [
    "video_id",
    "title",
    "channel",
    "published_at",
    "description",
    "url",
    "views",
    "likes",
    "comments",
    "duration_seconds",
    "transcript",
    "transcript_language",
    "transcript_source",
    "source_query",
    "fetched_at",
]

LEGACY_COLUMN_MAP = {
    "date": "published_at",
    "Video_description": "description",
    "Like count": "likes",
    "Video views": "views",
    "Comment count": "comments",
    "author": "channel",
    "guid": "url",
    "Language": "transcript_language",
}

TEXT_COLUMNS = [
    "video_id",
    "title",
    "channel",
    "published_at",
    "description",
    "url",
    "transcript",
    "transcript_language",
    "transcript_source",
    "source_query",
    "fetched_at",
]

NUMERIC_COLUMNS = ["views", "likes", "comments", "duration_seconds"]


def _blank_to_empty(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_dataset(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = df.copy()
    df.rename(columns=LEGACY_COLUMN_MAP, inplace=True)

    if df.columns.duplicated().any():
        deduplicated = pd.DataFrame()
        for column in dict.fromkeys(df.columns):
            matches = df.loc[:, df.columns == column]
            if isinstance(matches, pd.DataFrame):
                deduplicated[column] = matches.apply(
                    lambda row: next(
                        (
                            value
                            for value in row
                            if not pd.isna(value) and str(value).strip() != ""
                        ),
                        "",
                    ),
                    axis=1,
                )
            else:
                deduplicated[column] = matches
        df = deduplicated

    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in TEXT_COLUMNS:
        df[column] = df[column].map(_blank_to_empty)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["url"] = df.apply(
        lambda row: row["url"]
        or (
            f"https://www.youtube.com/watch?v={row['video_id']}"
            if row["video_id"]
            else ""
        ),
        axis=1,
    )

    df["transcript_length"] = df["transcript"].str.len()
    df["has_transcript"] = df["transcript_length"] > 0

    # Keep the strongest version of each row when legacy and new schemas overlap.
    df.sort_values(
        by=["has_transcript", "transcript_length", "views", "fetched_at"],
        ascending=[False, False, False, False],
        inplace=True,
        kind="stable",
    )

    df = df[df["video_id"] != ""]
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    df = df[STANDARD_COLUMNS + ["transcript_length", "has_transcript"]]

    df.sort_values(by=["published_at", "views", "title"], ascending=[False, False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=STANDARD_COLUMNS + ["transcript_length", "has_transcript"])

    df = pd.read_excel(path)
    return normalize_dataset(df)


def save_dataset(df: pd.DataFrame, path: Path | str = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_dataset(df)
    try:
        normalized[STANDARD_COLUMNS].to_excel(path, index=False)
    except PermissionError as exc:
        backup_path = path.with_name(f"{path.stem}.autosave{path.suffix}")
        normalized[STANDARD_COLUMNS].to_excel(backup_path, index=False)
        raise RuntimeError(
            f"Could not write to '{path}' because it is open in another program. "
            f"Close the file and try again. A backup copy was saved to '{backup_path}'."
        ) from exc
    return normalized


def save_new_data(new_data, path: Path | str = DATA_PATH) -> pd.DataFrame:
    df_new = normalize_dataset(pd.DataFrame(new_data))
    df_old = load_dataset(path)
    combined = pd.concat([df_old[STANDARD_COLUMNS], df_new[STANDARD_COLUMNS]], ignore_index=True)
    return save_dataset(combined, path)


def get_dataset_summary(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty:
        return {
            "total_videos": 0,
            "videos_with_transcripts": 0,
            "transcript_coverage_pct": 0.0,
            "total_views": 0,
        }

    videos_with_transcripts = int(df["has_transcript"].sum())
    return {
        "total_videos": int(len(df)),
        "videos_with_transcripts": videos_with_transcripts,
        "transcript_coverage_pct": round((videos_with_transcripts / len(df)) * 100, 1),
        "total_views": int(df["views"].fillna(0).sum()),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
