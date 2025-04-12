import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

# --- IMPORTANT SECURITY WARNING ---
# Hardcoding your API token directly in the code is generally NOT recommended,
# especially if you plan to share or deploy this application.
# It's safer to use environment variables or Streamlit secrets.
# ---

# Set your Hugging Face API token directly here
HUGGINGFACE_TOKEN = "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"  # Replace with your actual token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN

# --- Helper Functions ---

@st.cache_data
def get_transcript(video_id):
    """Fetches the transcript of a YouTube video."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en'])
        if not transcript:
            transcript = transcript_list.find_generated_transcript(['en'])
        if not transcript:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        if not transcript:
            return None
        transcript_data = transcript.fetch()
        full_transcript = " ".join([item['text'] for item in transcript_data])
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

@st.cache_data
def split_transcript(transcript, chunk_size=1000, chunk_overlap=100):
    """Splits the transcript into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(transcript)
    return chunks

@st.cache_resource
def create_vector_store(text_chunks):
    """Creates a Chroma vector store from the text chunks using HuggingFace embeddings."""
    try:
        model_name = "all-MiniLM-L6-v2"  # You can try other models
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store with HuggingFaceEmbeddings: {e}")
        return None

@st.cache_resource
def create_retrieval_qa_chain(vector_store, llm_model_name="google/flan-t5-small"):
    """Creates a RetrievalQA chain using a HuggingFace Hub LLM."""
    try:
        llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={"temperature": 0.2, "max_length": 512})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RetrievalQA chain with HuggingFaceHub: {e}")
        return None

# --- Main Application ---

def main():
    st.title("YouTube Video Chatbot (Free)")

    youtube_url = st.text_input("Enter the YouTube video URL:")

    if youtube_url:
        try:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        except IndexError:
            st.error("Invalid YouTube URL")
            return

        with st.spinner("Fetching and processing transcript..."):
            transcript_text = get_transcript(video_id)

            if transcript_text:
                text_chunks = split_transcript(transcript_text)
                vector_store = create_vector_store(text_chunks)
                if vector_store:  # Only proceed if vector store was created successfully
                    st.session_state.qa_chain = create_retrieval_qa_chain(vector_store)
                    if "qa_chain" in st.session_state:
                        st.success("Transcript processed. You can now ask questions!")
                    else:
                        st.error("Failed to create the question answering chain.")
                else:
                    st.error("Failed to create the vector store.")
            else:
                st.error("Could not retrieve or process the transcript for this video.")

    if "qa_chain" in st.session_state:
        st.subheader("Ask a question about the video:")
        query = st.text_input("Your question:")
        if query:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    st.write("Answer:", result["result"])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.error("Please ensure your Hugging Face API token is correctly set if the model requires it.")

if __name__ == "__main__":
    main()
