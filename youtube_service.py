import os
import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
import streamlit as st
from youtube_transcript import transcript_fetch

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY in .env")

# Build YouTube client
youtube = build("youtube", "v3", developerKey=API_KEY)


# ---------------- SEARCH ---------------- #
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

            if video_id:
                snippet = item.get("snippet", {})
                videos.append({
                    "video_id": video_id,
                    "title": snippet.get("title"),
                    "channel": snippet.get("channelTitle"),
                    "published_at": snippet.get("publishedAt"),
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        remaining -= batch_size

    return videos


# ---------------- ENRICH ---------------- #
@st.cache_data(show_spinner=False)
def enrich_videos(video_list):

    video_ids = [v["video_id"] for v in video_list]

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
                "duration_seconds": duration_seconds,
                "transcript":transcript_fetch(vid)
            }

        # Efficient merge
        video_dict = {v["video_id"]: v for v in video_list}

        for vid, stats in stats_map.items():
            if vid in video_dict:
                video_dict[vid].update(stats)

        video_list = list(video_dict.values())

    return video_list
