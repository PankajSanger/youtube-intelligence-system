from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import isodate
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    st.error("Missing YOUTUBE_API_KEY in .env")
    st.stop()

youtube = build("youtube", "v3", developerKey=API_KEY)

st.set_page_config(page_title="YouTube Platform Analyzer", layout="wide")

# ---------------- SEARCH SERVICE ---------------- #

@st.cache_data(show_spinner=False)
def search_videos(query, max_results, published_after, published_before, order):

    videos = []
    next_page_token = None
    remaining = max_results

    while remaining > 0:
        batch_size = min(50, remaining)

        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=batch_size,
            order=order,
            publishedAfter=published_after,
            publishedBefore=published_before,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response.get("items", []):
            video_id = item["id"].get("videoId")
            if video_id:
                videos.append({
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        remaining -= batch_size

    return videos


# ---------------- ENRICHMENT SERVICE ---------------- #

@st.cache_data(show_spinner=False)
def enrich_videos(video_list):

    video_ids = [v["video_id"] for v in video_list]

    enriched_data = []

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]

        request = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(batch_ids)
        )

        response = request.execute()

        stats_map = {}

        for item in response.get("items", []):
            vid = item["id"]

            duration = item["contentDetails"]["duration"]
            duration_seconds = int(isodate.parse_duration(duration).total_seconds())

            stats_map[vid] = {
                "views": int(item["statistics"].get("viewCount", 0)),
                "likes": int(item["statistics"].get("likeCount", 0)),
                "comments": int(item["statistics"].get("commentCount", 0)),
                "duration_seconds": duration_seconds
            }

        for video in video_list:
            vid = video["video_id"]
            if vid in stats_map:
                video.update(stats_map[vid])

    return video_list


# ---------------- STREAMLIT UI ---------------- #

st.title("🎥 YouTube Platform Analyzer ")

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

        # Derived Metrics
        df["engagement_rate"] = (
            (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
        ) * 100

        st.success(f"Analyzed {len(df)} videos")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "youtube_advanced_analysis.csv",
            "text/csv"
        )

    except HttpError as e:
        st.error(f"YouTube API Error: {e}")