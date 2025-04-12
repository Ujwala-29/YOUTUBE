import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp  # or use HuggingFacePipeline
import os
import re

# Set the title of the web app
st.title("🎥 YouTube Video Chatbot (Free & Local)")

# Input fields for YouTube URL and Question
video_url = st.text_input("Enter YouTube Video URL")
question = st.text_input("Ask a question about the video")

# Set model path for GGUF model
MODEL_PATH = "models/llama-2-7b-chat.gguf"  # Update with your model path

if not os.path.exists(MODEL_PATH):
    st.warning("Download a GGUF model (e.g., LLaMA 2 7B) and place it in the `models` folder.")
    st.stop()

@st.cache_resource
def load_llm():
    # Load LlamaCpp model for local inference
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        temperature=0.1,
        top_p=0.95,
        verbose=False,
    )

def get_video_id(url):
    # Extract video ID from the URL
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    # Fetch transcript of the video using YouTube Transcript API
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([x["text"] for x in transcript])
    return full_text

if video_url and question:
    with st.spinner("Processing video..."):
        # Extract video ID from URL
        video_id = get_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Could not extract video ID.")
            st.stop()

        # Get the transcript of the video
        raw_text = get_transcript(video_id)

        # Split text into chunks for processing
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([raw_text])

        # Create embeddings and vector store using FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Load language model
        llm = load_llm()

        # Set up RetrievalQA chain to answer questions based on video content
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False,
        )

        # Get the answer for the question
        answer = qa.run(question)

    # Show the answer to the user
    st.success("Answer:")
    st.write(answer)
