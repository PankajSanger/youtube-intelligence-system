import streamlit as st
import pandas as pd
from googleapiclient.errors import HttpError
from utils.data_manager import save_new_data
from youtube_service import search_videos, enrich_videos

# RAG
from rag.retriever import load_or_create_index
from rag.pipeline import run_query

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="YouTube Platform Analyzer", layout="wide")

st.title("🎥 YouTube Platform Analyzer")

DATA_PATH = "data/youtube_data_for_project.xlsx"


# ======================
# SESSION STATE INIT
# ======================
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None


# ======================
# ⚙️ INDEX SETUP SECTION
# ======================
st.subheader("⚙️ Setup (One-Time)")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Build / Load FAISS Index"):

        with st.spinner("Processing transcripts, building embeddings, creating index... ⏳"):

            vectorstore = load_or_create_index(DATA_PATH)

            st.session_state["vectorstore"] = vectorstore

        st.success("✅ FAISS Index Ready!")

with col2:
    if st.button("🗑️ Reset Index (Rebuild)"):

        import shutil
        import os

        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        st.session_state["vectorstore"] = None

        st.warning("Index deleted. Rebuild required.")


# ======================
# LOAD VECTORSTORE FROM SESSION
# ======================
vectorstore = st.session_state.get("vectorstore")


# ======================
# 📊 YOUTUBE ANALYSIS UI
# ======================
st.divider()
st.subheader("📊 YouTube Data Fetching")

query = st.text_input("Enter Query Keywords")

col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("Start Date")

with col2:
    end_date = st.date_input("End Date")

with col3:
    order = st.selectbox("Sort By", ["date", "viewCount", "relevance"])

max_results = st.slider("Max Results", 10, 300, 50)


if st.button("🔍 Run Analysis"):

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    after = start_date.strftime("%Y-%m-%dT00:00:00Z")
    before = end_date.strftime("%Y-%m-%dT23:59:59Z")

    try:
        with st.spinner("Fetching videos..."):
            base_videos = search_videos(query, max_results, after, before, order)

        if not base_videos:
            st.info("No videos found.")
            st.stop()

        with st.spinner("Enriching data (this may take time due to transcripts)..."):
            enriched = enrich_videos(base_videos)

        df = save_new_data(enriched)

        st.success("✅ Data saved to dataset!")

        st.warning("⚠️ New data added. Please rebuild index from top section.")

        df["engagement_rate"] = (
            (df["likes"] + df["comments"]) / df["views"].replace(0, 1)
        ) * 100

        st.success(f"✅ Analyzed {len(df)} videos")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "youtube_analysis.csv",
            "text/csv"
        )

    except HttpError as e:
        st.error(f"YouTube API Error: {e}")


# ======================
# 🔍 RAG QUESTION ANSWERING
# ======================
st.divider()
st.subheader("🔍 Ask Questions from Transcripts")

user_query = st.text_input("Ask your question about videos")


if user_query:

    if vectorstore is None:
        st.warning("⚠️ Please build the FAISS index first (top section)")
        st.stop()

    with st.spinner("Analyzing transcripts... 🤖"):
        answer = run_query(vectorstore, user_query)

    st.markdown("### 🧠 Answer")
    st.write(answer)


# ======================
# FOOTER STATUS
# ======================
st.divider()

if vectorstore is None:
    st.info("ℹ️ Index not loaded. Please build it to enable AI querying.")
else:
    st.success("✅ AI system ready for queries")