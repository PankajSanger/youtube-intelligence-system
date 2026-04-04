import os
import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
import streamlit as st

from youtube_transcript import transcript_fetch

# ======================
# CONFIG
# ======================
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("❌ Missing YOUTUBE_API_KEY in .env")

youtube = build("youtube", "v3", developerKey=API_KEY)


# ======================
# SEARCH VIDEOS
# ======================
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
            video_id = item.get("id", {}).get("videoId")

            if not video_id:
                continue

            snippet = item.get("snippet", {})

            videos.append({
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", "")
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

        remaining -= batch_size

    return videos


# ======================
# ENRICH VIDEOS
# ======================
@st.cache_data(show_spinner=False)
def enrich_videos(video_list):

    if not video_list:
        return []

    video_ids = [v["video_id"] for v in video_list]

    stats_map = {}

    # Batch processing (YouTube API limit = 50)
    for i in range(0, len(video_ids), 50):

        batch_ids = video_ids[i:i + 50]

        request = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(batch_ids)
        )

        response = request.execute()

        for item in response.get("items", []):

            vid = item["id"]

            # Duration parsing
            duration = item["contentDetails"].get("duration", "PT0S")

            try:
                duration_seconds = int(isodate.parse_duration(duration).total_seconds())
            except:
                duration_seconds = 0

            stats_map[vid] = {
                "views": int(item["statistics"].get("viewCount", 0)),
                "likes": int(item["statistics"].get("likeCount", 0)),
                "comments": int(item["statistics"].get("commentCount", 0)),
                "duration_seconds": duration_seconds
            }

    # Merge stats + transcript
    enriched_videos = []

    for video in video_list:

        vid = video["video_id"]

        stats = stats_map.get(vid, {})

        # Fetch transcript (safe)
        transcript = transcript_fetch(vid)

        enriched_videos.append({
            **video,
            "views": stats.get("views", 0),
            "likes": stats.get("likes", 0),
            "comments": stats.get("comments", 0),
            "duration_seconds": stats.get("duration_seconds", 0),
            "transcript": transcript
        })

    return enriched_videos