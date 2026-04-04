from __future__ import annotations

from apify_service import fetch_transcript_with_apify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    NotTranslatable,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
)

TRANSCRIPT_API = YouTubeTranscriptApi()
PREFERRED_LANGUAGES = ("en", "en-US", "hi", "hi-IN")


def _join_segments(transcript) -> str:
    try:
        return " ".join(segment.text for segment in transcript if getattr(segment, "text", "")).strip()
    except Exception:
        return ""


def _fetch_direct(video_id: str):
    transcript = TRANSCRIPT_API.fetch(video_id, languages=PREFERRED_LANGUAGES, preserve_formatting=False)
    text = _join_segments(transcript)
    language_code = getattr(transcript, "language_code", "")
    is_generated = bool(getattr(transcript, "is_generated", False))
    source = "generated" if is_generated else "manual"
    return {
        "transcript": text,
        "transcript_language": language_code,
        "transcript_source": source,
    }


def transcript_fetch(video_id: str) -> dict[str, str]:
    """
    Fetch transcript metadata with English preference and translation fallback.
    """

    empty = {
        "transcript": "",
        "transcript_language": "",
        "transcript_source": "unavailable",
    }

    try:
        direct = _fetch_direct(video_id)
        if direct["transcript"]:
            return direct
    except NoTranscriptFound:
        pass
    except (TranscriptsDisabled, VideoUnavailable, CouldNotRetrieveTranscript):
        return empty
    except Exception:
        pass

    try:
        transcript_list = TRANSCRIPT_API.list(video_id)
        preferred = transcript_list.find_transcript(PREFERRED_LANGUAGES)
        fetched = preferred.fetch(preserve_formatting=False)
        text = _join_segments(fetched)
        if text:
            source = "generated" if getattr(preferred, "is_generated", False) else "manual"
            return {
                "transcript": text,
                "transcript_language": getattr(preferred, "language_code", ""),
                "transcript_source": source,
            }
    except NoTranscriptFound:
        pass
    except (TranscriptsDisabled, VideoUnavailable, CouldNotRetrieveTranscript):
        return empty
    except Exception:
        pass

    try:
        transcript_list = TRANSCRIPT_API.list(video_id)
        for transcript in transcript_list:
            if not getattr(transcript, "is_translatable", False):
                continue
            try:
                translated = transcript.translate("en")
                fetched = translated.fetch(preserve_formatting=False)
                text = _join_segments(fetched)
                if text:
                    source = "translated-generated" if getattr(transcript, "is_generated", False) else "translated-manual"
                    return {
                        "transcript": text,
                        "transcript_language": "en",
                        "transcript_source": source,
                    }
            except (NotTranslatable, TranslationLanguageNotAvailable):
                continue
            except Exception:
                continue
    except Exception:
        return empty

    apify_result = fetch_transcript_with_apify(video_id)
    if apify_result.get("transcript", "").strip():
        return apify_result

    return empty
