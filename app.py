from __future__ import annotations

import os
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from googleapiclient.errors import HttpError

from apify_service import apify_is_configured, fetch_transcript_with_apify
from openai_service import openai_is_configured
from rag.pipeline import run_query
from rag.retriever import load_or_create_index
from utils.data_manager import DATA_PATH, get_dataset_summary, load_dataset, save_new_data
from youtube_service import enrich_videos, search_videos

INDEX_PATH = Path("faiss_index")

st.set_page_config(page_title="YouTube Intelligence Studio", page_icon="YT", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fc;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            border-radius: 12px;
            border: none;
            font-weight: 700;
            box-shadow: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 999px;
            padding: 0 1rem;
            background: white;
            border: 1px solid #d8dee8;
        }

        .stTabs [aria-selected="true"] {
            background: #111827;
            color: white !important;
            border-color: #111827;
        }

        .video-card {
            background: white;
            border: 1px solid #d8dee8;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 1rem;
        }

        .video-thumb {
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            display: block;
            background: #e5e7eb;
        }

        .video-body {
            padding: 0.9rem;
        }

        .video-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.35;
            margin-bottom: 0.45rem;
            min-height: 2.6rem;
        }

        .video-meta {
            color: #667085;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 0.65rem;
        }

        .video-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.6rem;
        }

        .tag {
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            background: #f3f4f6;
            color: #374151;
        }

        .tag-ok {
            background: #e8f7ee;
            color: #157347;
        }

        .tag-warn {
            background: #fdecec;
            color: #b42318;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float | int) -> str:
    value = int(value or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_duration(seconds: float | int) -> str:
    seconds = int(seconds or 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def compute_engagement_rate(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    views = pd.to_numeric(df["views"], errors="coerce").fillna(0).replace(0, 1)
    likes = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    comments = pd.to_numeric(df["comments"], errors="coerce").fillna(0)
    return ((likes + comments) / views) * 100


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display_df = df.copy()
    display_df["engagement_rate"] = compute_engagement_rate(display_df).round(2)
    for column in ["views", "likes", "comments", "duration_seconds"]:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce").fillna(0).astype(int)
    return display_df[
        [
            "video_id",
            "title",
            "channel",
            "published_at",
            "views",
            "likes",
            "comments",
            "duration_seconds",
            "transcript_language",
            "transcript_source",
            "engagement_rate",
            "url",
        ]
    ]


def top_channels(df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["channel", "videos", "total_views"])
    grouped = (
        df.groupby("channel", dropna=False)
        .agg(videos=("video_id", "count"), total_views=("views", "sum"))
        .reset_index()
        .sort_values(["videos", "total_views"], ascending=[False, False])
        .head(limit)
    )
    grouped["channel"] = grouped["channel"].replace("", "Unknown channel")
    grouped["total_views"] = pd.to_numeric(grouped["total_views"], errors="coerce").fillna(0).astype(int)
    return grouped


def render_video_card(row: pd.Series) -> None:
    transcript_text = str(row.get("transcript", "") or "").strip()
    transcript_badge = (
        '<span class="tag tag-ok">Transcript ready</span>'
        if transcript_text
        else '<span class="tag tag-warn">No transcript</span>'
    )
    title = str(row.get("title", "") or "Untitled")
    channel = str(row.get("channel", "") or "Unknown channel")
    published_at = str(row.get("published_at", "") or "Unknown date")
    url = str(row.get("url", "") or f"https://www.youtube.com/watch?v={row.get('video_id', '')}")
    language = str(row.get("transcript_language", "") or "").strip() or "n/a"

    st.markdown(
        f"""
        <div class="video-card">
            <img class="video-thumb" src="{thumbnail_url(str(row.get('video_id', '')))}" alt="thumbnail" />
            <div class="video-body">
                <div class="video-title">{title}</div>
                <div class="video-meta">{channel}<br>{published_at}</div>
                <div class="video-tags">
                    <span class="tag">{format_number(row.get('views', 0))} views</span>
                    <span class="tag">{format_duration(row.get('duration_seconds', 0))}</span>
                    <span class="tag">Lang: {language}</span>
                    {transcript_badge}
                </div>
                <a href="{url}" target="_blank">Open on YouTube</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_video_filters(
    videos: list[dict],
    min_duration_seconds: int,
    require_transcript: bool,
) -> tuple[list[dict], dict[str, int]]:
    duration_removed = 0
    transcript_removed = 0
    kept: list[dict] = []

    for video in videos:
        duration = int(video.get("duration_seconds", 0) or 0)
        has_transcript = bool(str(video.get("transcript", "") or "").strip())

        if duration < min_duration_seconds:
            duration_removed += 1
            continue
        if require_transcript and not has_transcript:
            transcript_removed += 1
            continue
        kept.append(video)

    return kept, {
        "fetched": len(videos),
        "kept": len(kept),
        "duration_removed": duration_removed,
        "transcript_removed": transcript_removed,
    }


inject_styles()

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "last_fetch_df" not in st.session_state:
    st.session_state["last_fetch_df"] = pd.DataFrame()
if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = ""

dataset_df = load_dataset(DATA_PATH)
summary = get_dataset_summary(dataset_df)
index_exists = INDEX_PATH.exists()

st.title("YouTube Intelligence Studio")
st.caption("Search videos, save results, and ask questions from transcripts.")

action_cols = st.columns([1, 1, 2], gap="large")
with action_cols[0]:
    if st.button("Build Index", width="stretch", type="primary"):
        try:
            with st.spinner("Preparing transcript chunks and embeddings..."):
                st.session_state["vectorstore"] = load_or_create_index(DATA_PATH)
            st.success("Index is ready.")
        except Exception as exc:
            st.session_state["vectorstore"] = None
            st.error(f"Index setup failed: {exc}")
with action_cols[1]:
    if st.button("Reset Index", width="stretch"):
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        st.session_state["vectorstore"] = None
        st.warning("Saved index removed.")
with action_cols[2]:
    status_parts = [
        "Index ready" if index_exists else "Index missing",
        "YouTube key ready" if os.getenv("YOUTUBE_API_KEY") else "YouTube key missing",
        "Apify fallback ready" if apify_is_configured() else "Apify fallback not configured",
        "OpenAI ready" if openai_is_configured() else "OpenAI missing",
    ]
    st.caption(" | ".join(status_parts))

metric_cols = st.columns(4, gap="large")
metric_cols[0].metric("Library Size", f"{summary['total_videos']}")
metric_cols[1].metric("Transcript Coverage", f"{summary['transcript_coverage_pct']}%")
metric_cols[2].metric("Transcript-Ready", f"{summary['videos_with_transcripts']}")
metric_cols[3].metric("Tracked Views", f"{summary['total_views']:,}")

tabs = st.tabs(["Collect", "Review", "Library", "Ask AI"])

with tabs[0]:
    st.subheader("Search Videos")
    with st.form("analysis_form", clear_on_submit=False):
        query = st.text_input("Query keywords", placeholder="Example: herbal hair oil tutorial")
        row1 = st.columns(3)
        with row1[0]:
            start_date = st.date_input("Start date", value=date.today() - timedelta(days=30))
        with row1[1]:
            end_date = st.date_input("End date", value=date.today())
        with row1[2]:
            order = st.selectbox("Sort by", ["date", "viewCount", "relevance"])

        row2 = st.columns(3)
        with row2[0]:
            max_results = st.slider("Max results", 10, 200, 50, 10)
        with row2[1]:
            min_duration_seconds = st.slider("Minimum duration", 0, 900, 60, 30)
        with row2[2]:
            require_transcript = st.checkbox("Only save transcript videos", value=False)

        run_analysis = st.form_submit_button("Run Analysis", width="stretch")

    if run_analysis:
        if not query.strip():
            st.warning("Enter a search query before running analysis.")
        elif start_date > end_date:
            st.error("Start date cannot be after end date.")
        else:
            after = start_date.strftime("%Y-%m-%dT00:00:00Z")
            before = end_date.strftime("%Y-%m-%dT23:59:59Z")
            try:
                with st.spinner("Fetching videos from YouTube..."):
                    base_videos = search_videos(query, max_results, after, before, order)

                if not base_videos:
                    st.info("No videos were found for this query and date range.")
                else:
                    with st.spinner("Enriching videos with stats and transcripts..."):
                        enriched_videos = enrich_videos(base_videos)

                    filtered_videos, filter_stats = apply_video_filters(
                        enriched_videos,
                        min_duration_seconds=min_duration_seconds,
                        require_transcript=require_transcript,
                    )

                    if not filtered_videos and min_duration_seconds == 0 and not require_transcript and enriched_videos:
                        filtered_videos = enriched_videos
                        filter_stats["kept"] = len(filtered_videos)
                        filter_stats["duration_removed"] = 0
                        filter_stats["transcript_removed"] = 0

                    if not filtered_videos:
                        st.warning(
                            "No videos passed the current filters. "
                            f"Fetched: {filter_stats['fetched']}, "
                            f"removed by duration: {filter_stats['duration_removed']}, "
                            f"removed by transcript: {filter_stats['transcript_removed']}."
                        )
                    else:
                        dataset_df = save_new_data(filtered_videos)
                        st.session_state["last_fetch_df"] = pd.DataFrame(filtered_videos)
                        summary = get_dataset_summary(dataset_df)
                        st.success(f"Saved {len(filtered_videos)} videos.")
                        if filter_stats["fetched"] != filter_stats["kept"]:
                            st.info(
                                f"Fetched {filter_stats['fetched']} videos. "
                                f"Filtered out {filter_stats['duration_removed']} by duration and "
                                f"{filter_stats['transcript_removed']} by transcript requirement."
                            )
            except HttpError as exc:
                st.error(f"YouTube API error: {exc}")
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

with tabs[1]:
    st.subheader("Latest Results")
    latest_fetch_df = st.session_state["last_fetch_df"]
    if latest_fetch_df.empty:
        st.info("No results yet.")
    else:
        latest_cards = latest_fetch_df.head(6).reset_index(drop=True)
        for start in range(0, len(latest_cards), 3):
            cols = st.columns(3, gap="large")
            for col, idx in zip(cols, range(start, min(start + 3, len(latest_cards)))):
                with col:
                    render_video_card(latest_cards.iloc[idx])

        st.dataframe(format_dataframe_for_display(latest_fetch_df), width="stretch", hide_index=True)
        st.download_button(
            "Download Latest Results",
            format_dataframe_for_display(latest_fetch_df).to_csv(index=False),
            "youtube_analysis_latest.csv",
            "text/csv",
        )

with tabs[2]:
    st.subheader("Library")
    if dataset_df.empty:
        st.info("No saved videos yet.")
    else:
        top_cols = st.columns([1, 1], gap="large")
        with top_cols[0]:
            st.info(f"Saved videos: {summary['total_videos']} | Transcript-ready: {summary['videos_with_transcripts']}")
        with top_cols[1]:
            channels_df = top_channels(dataset_df)
            if not channels_df.empty:
                st.dataframe(channels_df, width="stretch", hide_index=True)

        library_cards = dataset_df.head(6).reset_index(drop=True)
        for start in range(0, len(library_cards), 3):
            cols = st.columns(3, gap="large")
            for col, idx in zip(cols, range(start, min(start + 3, len(library_cards)))):
                with col:
                    render_video_card(library_cards.iloc[idx])

        st.dataframe(format_dataframe_for_display(dataset_df.head(25)), width="stretch", hide_index=True)
        st.download_button(
            "Download Full Dataset",
            dataset_df.to_csv(index=False),
            "youtube_dataset.csv",
            "text/csv",
        )

with tabs[3]:
    st.subheader("Ask AI")
    user_query = st.text_area(
        "Question",
        placeholder="Which videos mention ingredients, preparation steps, or comparisons?",
        height=130,
    )
    if st.button("Generate Answer", width="stretch", type="primary"):
        if not user_query.strip():
            st.warning("Enter a question first.")
        elif st.session_state["vectorstore"] is None:
            st.warning("Build the FAISS index first.")
        else:
            with st.spinner("Searching transcript context..."):
                st.session_state["last_answer"] = run_query(st.session_state["vectorstore"], user_query)

    if st.session_state["last_answer"]:
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown("#### Answer")
        st.write(st.session_state["last_answer"])
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
with st.expander("Transcript Debug Tool"):
    debug_video_id = st.text_input("Video ID for transcript test", placeholder="Example: dQw4w9WgXcQ")
    if st.button("Test Apify Transcript", width="stretch"):
        if not debug_video_id.strip():
            st.warning("Enter a video ID first.")
        else:
            with st.spinner("Calling Apify transcript actor..."):
                debug_result = fetch_transcript_with_apify(debug_video_id.strip())
            st.write(
                {
                    "transcript_source": debug_result.get("transcript_source", ""),
                    "transcript_language": debug_result.get("transcript_language", ""),
                    "transcript_length": len(debug_result.get("transcript", "")),
                }
            )
            if debug_result.get("debug_error"):
                st.code(debug_result["debug_error"])
            preview = debug_result.get("transcript", "")[:1200]
            if preview:
                st.text_area("Transcript preview", value=preview, height=220)
            else:
                st.info("No transcript text returned by Apify.")
