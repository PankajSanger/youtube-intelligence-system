from __future__ import annotations

import os
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from googleapiclient.errors import HttpError

from rag.pipeline import run_query
from rag.retriever import load_or_create_index
from utils.data_manager import DATA_PATH, get_dataset_summary, load_dataset, save_new_data
from youtube_service import enrich_videos, search_videos

INDEX_PATH = Path("faiss_index")

st.set_page_config(page_title="YouTube Platform Analyzer", layout="wide")
st.title("YouTube Platform Analyzer")
st.caption("Search YouTube, save structured video data, and ask transcript-aware questions from one workspace.")

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "last_fetch_df" not in st.session_state:
    st.session_state["last_fetch_df"] = pd.DataFrame()

dataset_df = load_dataset(DATA_PATH)
summary = get_dataset_summary(dataset_df)


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


metric_cols = st.columns(4)
metric_cols[0].metric("Videos in Dataset", f"{summary['total_videos']}")
metric_cols[1].metric("Transcript Coverage", f"{summary['transcript_coverage_pct']}%")
metric_cols[2].metric("Videos With Transcripts", f"{summary['videos_with_transcripts']}")
metric_cols[3].metric("Total Views Tracked", f"{summary['total_views']:,}")

st.divider()
st.subheader("Index Setup")

setup_col1, setup_col2 = st.columns(2)

with setup_col1:
    if st.button("Build or Load FAISS Index", width="stretch"):
        try:
            with st.spinner("Preparing transcript chunks and embeddings..."):
                st.session_state["vectorstore"] = load_or_create_index(DATA_PATH)
            st.success("FAISS index is ready.")
        except Exception as exc:
            st.session_state["vectorstore"] = None
            st.error(f"Index setup failed: {exc}")

with setup_col2:
    if st.button("Reset Index", width="stretch"):
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        st.session_state["vectorstore"] = None
        st.warning("Index deleted. Rebuild it when you want transcript search again.")

if INDEX_PATH.exists():
    st.caption(f"Saved index detected at `{INDEX_PATH}`.")
else:
    st.caption("No saved index found yet.")

st.divider()
st.subheader("YouTube Data Fetching")

with st.form("analysis_form"):
    query = st.text_input("Query keywords", placeholder="Example: herbal hair oil")

    input_col1, input_col2, input_col3 = st.columns(3)
    with input_col1:
        start_date = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with input_col2:
        end_date = st.date_input("End date", value=date.today())
    with input_col3:
        order = st.selectbox("Sort by", ["date", "viewCount", "relevance"])

    max_results = st.slider("Max results", min_value=10, max_value=200, value=50, step=10)
    quality_col1, quality_col2 = st.columns(2)
    with quality_col1:
        min_duration_seconds = st.slider("Minimum duration (seconds)", min_value=0, max_value=600, value=60, step=30)
    with quality_col2:
        require_transcript = st.checkbox("Only save videos with transcripts", value=False)
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

                filtered_videos = [
                    video
                    for video in enriched_videos
                    if video.get("duration_seconds", 0) >= min_duration_seconds
                    and (not require_transcript or video.get("transcript", "").strip())
                ]

                removed_count = len(enriched_videos) - len(filtered_videos)
                if not filtered_videos:
                    st.warning(
                        "No videos passed the quality filters. Try lowering the minimum duration or disabling transcript-only mode."
                    )
                    st.stop()

                dataset_df = save_new_data(filtered_videos)
                st.session_state["last_fetch_df"] = pd.DataFrame(filtered_videos)
                summary = get_dataset_summary(dataset_df)

                st.success(f"Saved {len(filtered_videos)} fetched videos. Dataset now contains {len(dataset_df)} unique videos.")
                if removed_count:
                    st.info(f"Filtered out {removed_count} videos that did not meet the duration or transcript criteria.")

                latest_df = format_dataframe_for_display(pd.DataFrame(filtered_videos))
                st.markdown("### Latest Fetch")
                st.dataframe(latest_df, width="stretch")

                st.download_button(
                    "Download Latest Fetch as CSV",
                    latest_df.to_csv(index=False),
                    "youtube_analysis_latest.csv",
                    "text/csv",
                )

                if summary["videos_with_transcripts"] == 0:
                    st.warning("No transcripts are available yet, so the RAG index cannot answer questions meaningfully.")
                elif summary["transcript_coverage_pct"] < 25:
                    st.warning("Transcript coverage is still low. RAG answers may be limited until more transcript-enabled videos are collected.")

                st.info("If you added new data, rebuild the FAISS index above so question answering uses the latest transcripts.")

        except HttpError as exc:
            st.error(f"YouTube API error: {exc}")
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

st.divider()
st.subheader("Dataset Snapshot")

if dataset_df.empty:
    st.info("No dataset saved yet. Run an analysis to start collecting videos.")
else:
    dataset_display = format_dataframe_for_display(dataset_df.head(20))
    st.dataframe(dataset_display, width="stretch")
    st.download_button(
        "Download Full Dataset as CSV",
        dataset_df.to_csv(index=False),
        "youtube_dataset.csv",
        "text/csv",
    )

st.divider()
st.subheader("Ask Questions from Transcripts")

user_query = st.text_input("Ask a question about the indexed videos", placeholder="Which videos talk about ingredients and preparation steps?")
if user_query:
    if st.session_state["vectorstore"] is None:
        st.warning("Build or load the FAISS index first.")
    else:
        with st.spinner("Searching transcript context..."):
            answer = run_query(st.session_state["vectorstore"], user_query)
        st.markdown("### Answer")
        st.write(answer)

st.divider()
env_checks = []
env_checks.append("YOUTUBE_API_KEY configured" if os.getenv("YOUTUBE_API_KEY") else "YOUTUBE_API_KEY missing")
env_checks.append(
    "HUGGINGFACEHUB_API_TOKEN configured"
    if os.getenv("HUGGINGFACEHUB_API_TOKEN")
    else "HUGGINGFACEHUB_API_TOKEN missing (fallback transcript excerpts will still work)"
)
st.caption(" | ".join(env_checks))
