import streamlit as st
import pandas as pd
from googleapiclient.errors import HttpError
from youtube_service import search_videos, enrich_videos

st.set_page_config(page_title="YouTube Platform Analyzer", layout="wide")

st.title("🎥 YouTube Platform Analyzer")

query = st.text_input("Enter Query Keywords")

col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("Start Date")

with col2:
    end_date = st.date_input("End Date")

with col3:
    order = st.selectbox("Sort By", ["date", "viewCount", "relevance"])

max_results = st.slider("Max Results", 10, 300, 50)

if st.button("Run Analysis"):

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    after = start_date.strftime("%Y-%m-%dT00:00:00Z")
    before = end_date.strftime("%Y-%m-%dT23:59:59Z")

    try:
        with st.spinner("Searching videos..."):
            base_videos = search_videos(query, max_results, after, before, order)

        if not base_videos:
            st.info("No videos found.")
            st.stop()

        with st.spinner("Enriching video statistics..."):
            enriched = enrich_videos(base_videos)

        df = pd.DataFrame(enriched)

        df["engagement_rate"] = (
            (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
        ) * 100

        st.success(f"Analyzed {len(df)} videos")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "youtube_analysis.csv",
            "text/csv"
        )

    except HttpError as e:
        st.error(f"YouTube API Error: {e}")