import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredURLLoader
import validators
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="YouTube and Website Summarizer", page_icon="📖")
st.title("📖 YouTube Video and Website Summarizer")
st.subheader("Summarize The Youtube Video and website")

st.text("Youtube video should have english caption turned on to work properly")

with st.sidebar:
    groq_api = st.text_input("Enter your Groq API Key", type="password")

url = st.text_input("Enter YouTube video or Website URL")

prompt_template = """
Summarize the following content in under 300 words:
content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    query = parse_qs(parsed_url.query)
    return query.get("v", [None])[0]

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry["text"] for entry in transcript])
    return full_text

if st.button("Summarize"):
    if not groq_api or not url:
        st.error("Please provide both Groq API key and YouTube URL.")
    elif not validators.url(url):
        st.error("Invalid URL.")
    else:
        try:
            if "youtube.com" in url or "youtu.be" in url:
                video_id = get_video_id(url)
                transcript_text = get_transcript(video_id)
                doc = [Document(page_content=transcript_text)]
            else:
                loader=UnstructuredURLLoader(urls=[url],ssl_verify=False)
                doc=loader.load()
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api)
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

            with st.spinner("Summarizing..."):
                summary = chain.run(doc)

            st.success(summary)

        except Exception as e:
            st.error(f"Error: {str(e)}")

   