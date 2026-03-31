from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

def transcript_fetch(video_id):
    try:
        # Try fetching transcript in preferred languages
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(
            video_id,
            languages=['en', 'hi']
        )

        snippets_list = [snippet.text for snippet in fetched_transcript]

        final_transcript = " ".join(snippets_list)

        return final_transcript

    except TranscriptsDisabled:
        return "Transcript is disabled for this video."

    except NoTranscriptFound:
        return "No transcript found in the specified languages."

    except VideoUnavailable:
        return "Video is unavailable."

    except Exception as e:
        return f"Error occurred: {str(e)}"