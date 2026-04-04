# YouTube Transcript Intelligence System

Streamlit app for collecting YouTube video data, enriching it with transcripts, storing a normalized dataset, and querying transcript content with FAISS-based retrieval.

## What It Does

- Searches YouTube videos by keyword, date range, and sort order
- Pulls video stats such as views, likes, comments, and duration
- Fetches transcripts with `youtube-transcript-api` and Apify fallback
- Translates non-English transcripts to English with OpenAI
- Stores everything in a normalized Excel dataset
- Builds a FAISS index using OpenAI embeddings
- Answers transcript questions with OpenAI

## Project Structure

- `app.py`: Streamlit interface
- `youtube_service.py`: YouTube API search and enrichment
- `youtube_transcript.py`: transcript retrieval and fallback
- `apify_service.py`: Apify actor integration for transcript fallback
- `openai_service.py`: shared OpenAI client and model settings
- `utils/data_manager.py`: dataset normalization, deduplication, summary helpers
- `rag/index_builder.py`: transcript preprocessing and FAISS index creation
- `rag/openai_embeddings.py`: OpenAI embeddings adapter
- `rag/retriever.py`: vector retrieval helpers
- `rag/pipeline.py`: question answering pipeline

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set:

- `YOUTUBE_API_KEY`
- `OPENAI_API_KEY`
- `APIFY_API_TOKEN` (optional)
- `APIFY_TRANSCRIPT_ACTOR` (optional, defaults to `akash9078~youtube-transcript-extractor`)

Use raw token values in `.env`, not full URLs containing `?token=...`.

4. Run the app:

```bash
streamlit run app.py
```

## Notes

- The dataset is saved to `data/youtube_data_for_project.xlsx`.
- The FAISS index is saved in `faiss_index/`.
- OpenAI is used for translation, embeddings, and question answering.
- Apify transcript fallback defaults to `akash9078~youtube-transcript-extractor` using the working `videoUrl` input shape.
