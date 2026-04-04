from __future__ import annotations

import os

import isodate
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build

from utils.data_manager import utc_now_iso
from youtube_transcript import transcript_fetch

load_dotenv()


def get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in .env")
    return build("youtube", "v3", developerKey=api_key)


@st.cache_data(show_spinner=False)
def search_videos(query, max_results, published_after, published_before, order):
    youtube = get_youtube_client()
    videos = []
    next_page_token = None
    remaining = max_results

    while remaining > 0:
        batch_size = min(50, remaining)
        response = (
            youtube.search()
            .list(
                part="snippet",
                q=query,
                type="video",
                maxResults=batch_size,
                order=order,
                publishedAfter=published_after,
                publishedBefore=published_before,
                pageToken=next_page_token,
            )
            .execute()
        )

        for item in response.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue

            snippet = item.get("snippet", {})
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "").strip(),
                    "channel": snippet.get("channelTitle", "").strip(),
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", "").strip(),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "source_query": query.strip(),
                }
            )

        remaining -= batch_size
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return videos


@st.cache_data(show_spinner=False)
def enrich_videos(video_list):
    if not video_list:
        return []

    youtube = get_youtube_client()
    video_ids = [video["video_id"] for video in video_list if video.get("video_id")]
    stats_map = {}

    for index in range(0, len(video_ids), 50):
        batch_ids = video_ids[index : index + 50]
        response = (
            youtube.videos()
            .list(
                part="statistics,contentDetails,snippet",
                id=",".join(batch_ids),
            )
            .execute()
        )

        for item in response.get("items", []):
            duration = item.get("contentDetails", {}).get("duration", "PT0S")
            try:
                duration_seconds = int(isodate.parse_duration(duration).total_seconds())
            except Exception:
                duration_seconds = 0

            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            stats_map[item["id"]] = {
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "duration_seconds": duration_seconds,
                "description": snippet.get("description", "").strip(),
                "channel": snippet.get("channelTitle", "").strip(),
                "published_at": snippet.get("publishedAt", ""),
            }

    enriched_videos = []
    fetched_at = utc_now_iso()

    for video in video_list:
        video_id = video["video_id"]
        stats = stats_map.get(video_id, {})
        transcript_payload = transcript_fetch(video_id)

        enriched_videos.append(
            {
                **video,
                "channel": stats.get("channel", video.get("channel", "")),
                "published_at": stats.get("published_at", video.get("published_at", "")),
                "description": stats.get("description", video.get("description", "")),
                "views": stats.get("views", 0),
                "likes": stats.get("likes", 0),
                "comments": stats.get("comments", 0),
                "duration_seconds": stats.get("duration_seconds", 0),
                "fetched_at": fetched_at,
                **transcript_payload,
            }
        )

    return enriched_videos
