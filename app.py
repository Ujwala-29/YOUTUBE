import streamlit as st
from pytube import YouTube
from pydub import AudioSegment
from transformers import pipeline
import tempfile
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Set your Hugging Face API token
HUGGINGFACE_TOKEN = "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"  # Replace with your real token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_TOKEN


# --- HELPER FUNCTIONS ---

@st.cache_data
def get_transcript(video_url):
    """Downloads YouTube audio and transcribes it using Whisper."""
    try:
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(filename=temp_audio.name)

        # Convert to WAV format
        audio = AudioSegment.from_file(temp_audio.name)
        wav_path = temp_audio.name.replace(".mp4", ".wav")
        audio.export(wav_path, format="wav")

        # Transcribe with Whisper
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        result = pipe(wav_path)

        return result["text"]
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None


@st.cache_data
def split_transcript(transcript, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(transcript)
    return chunks


@st.cache_resource
def create_vector_store(text_chunks):
    try:
        model_name = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


@st.cache_resource
def create_retrieval_qa_chain(vector_store, llm_model_name="google/flan-t5-small"):
    try:
        llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={"temperature": 0.2, "max_length": 512})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RetrievalQA chain: {e}")
        return None


# --- STREAMLIT APP ---

def main():
    st.title("🎙️ YouTube Video Chatbot (Whisper + HuggingFace)")

    youtube_url = st.text_input("Enter the YouTube video URL:")

    if youtube_url:
        with st.spinner("Transcribing and processing..."):
            transcript_text = get_transcript(youtube_url)

            if transcript_text:
                text_chunks = split_transcript(transcript_text)
                vector_store = create_vector_store(text_chunks)
                if vector_store:
                    st.session_state.qa_chain = create_retrieval_qa_chain(vector_store)
                    if "qa_chain" in st.session_state:
                        st.success("Transcript ready. Ask your questions!")
                    else:
                        st.error("Failed to initialize question-answering chain.")
                else:
                    st.error("Failed to create vector store.")
            else:
                st.error("Transcript generation failed.")

    if "qa_chain" in st.session_state:
        st.subheader("Ask something about the video:")
        query = st.text_input("Your question:")
        if query:
            with st.spinner("Generating answer..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    st.write("Answer:", result["result"])
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")


if __name__ == "__main__":
    main()
