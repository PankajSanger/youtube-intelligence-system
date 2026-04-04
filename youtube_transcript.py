from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)


def transcript_fetch(video_id):
    """
    Fetch transcript in preferred languages (English > Hindi fallback)
    """

    try:
        # Try English first
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    except NoTranscriptFound:
        try:
            # Fallback to Hindi
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        except Exception:
            return ""

    except (TranscriptsDisabled, VideoUnavailable):
        return ""

    except Exception as e:
        print(f"[Transcript Error] {video_id}: {e}")
        return ""

    # Convert to plain text
    try:
        full_text = " ".join([t["text"] for t in transcript])
        return full_text.strip()
    except:
        return ""