# YouTube Transcript Intelligence System

Streamlit app for collecting YouTube video data, enriching it with transcripts, storing a normalized dataset, and querying transcript content with FAISS-based retrieval.

## What It Does

- Searches YouTube videos by keyword, date range, and sort order
- Pulls video stats such as views, likes, comments, and duration
- Attempts transcript retrieval with English preference and translation fallback
- Stores everything in a normalized Excel dataset
- Builds a FAISS index from transcript text
- Answers transcript questions with a Hugging Face model when configured
- Falls back to relevant transcript excerpts when LLM generation is unavailable

## Project Structure

- `app.py`: Streamlit interface
- `youtube_service.py`: YouTube API search and enrichment
- `youtube_transcript.py`: transcript retrieval and translation fallback
- `utils/data_manager.py`: dataset normalization, deduplication, summary helpers
- `rag/index_builder.py`: transcript preprocessing and FAISS index creation
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
- `HUGGINGFACEHUB_API_TOKEN` (optional but recommended)

4. Run the app:

```bash
streamlit run app.py
```

## Notes

- The dataset is saved to `data/youtube_data_for_project.xlsx`.
- The FAISS index is saved in `faiss_index/`.
- If transcript coverage is low, question answering quality will also be low.
- Current LangChain dependencies may behave best on Python 3.11 or 3.12. Python 3.14 can show compatibility warnings in some environments.
