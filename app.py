import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp  # or use HuggingFacePipeline

import os

st.title("🎥 YouTube Video Chatbot (Free & Local)")

video_url = st.text_input("Enter YouTube Video URL")

question = st.text_input("Ask a question about the video")

# Load model path (you must download a GGUF model and set path here)
MODEL_PATH = "models/llama-2-7b-chat.gguf"  # Update path
if not os.path.exists(MODEL_PATH):
    st.warning("Download a GGUF model (e.g., LLaMA 2 7B) and place it in the `models` folder.")
    st.stop()

@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        temperature=0.1,
        top_p=0.95,
        verbose=False,
    )

def get_video_id(url):
    import re
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([x["text"] for x in transcript])
    return full_text

if video_url and question:
    with st.spinner("Processing video..."):
        video_id = get_video_id(video_url)
        raw_text = get_transcript(video_id)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = load_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False,
        )

        answer = qa.run(question)

    st.success("Answer:")
    st.write(answer)
