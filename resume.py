import os
import streamlit as st


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_core.runnables import RunnableLambda, RunnableMap
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
import re

# Load environment variables
load_dotenv()

# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create or load Chroma vector store
VECTOR_STORE_DIR = "chroma_store"

if os.path.exists(VECTOR_STORE_DIR):
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# Extract text from uploaded files
def extract_text_from_resume(file):

    temp_file_path = f"temp_{file.name}"
    
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# Store resume analysis in vector store
def store_resume_analysis(resume_text, analysis, doc_id):
    documents = split_text(analysis)
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
    vectorstore.persist()

# Extract percentage score from analysis text
def extract_suitability_score(text):
    match = re.search(r"Suitability Score: (\d{1,3})%", text)
    if match:
        return int(match.group(1))
    return None

# Main App
def main():
    st.set_page_config(page_title="Resume Screening App", layout="wide")
    st.title("Resume Screening with LCEL and Vector Store")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Job Requirements")
        job_requirements = st.text_area("Enter job requirements", height=300)
    with col2:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader("Upload a resume", type=["pdf", "docx", "txt"])

    if st.button("Analyze") and uploaded_file and job_requirements:
        with st.spinner("Processing..."):
            resume_text = extract_text_from_resume(uploaded_file)
            with st.expander("View Resume Text"):
                st.text(resume_text)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.2
            )

            prompt_template = PromptTemplate(
                input_variables=["job_requirements", "resume_text"],
                template="""
                You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

                Job Requirements:
                {job_requirements}

                Resume:
                {resume_text}

                Provide a structured analysis of how well the resume matches the job requirements. 
                At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well the resume aligns with the job.
                Format: Suitability Score: XX%
                """
            )

            chain = (
                RunnableMap({
                    "job_requirements": lambda x: x["job_requirements"],
                    "resume_text": lambda x: x["resume_text"]
                })
                | prompt_template
                | llm
                | StrOutputParser()
            )

            analysis = chain.invoke({
                "job_requirements": job_requirements,
                "resume_text": resume_text
            })

            st.header("AI Analysis")
            st.markdown(analysis)

            # Extract and display the suitability score
            suitability_score = extract_suitability_score(analysis)
            if suitability_score is not None:
                st.metric(label="Resume Suitability Score", value=f"{suitability_score}%")
            else:
                st.warning("Analysis Done.")

            # Store analysis in vector DB
            store_resume_analysis(resume_text, analysis, os.path.splitext(uploaded_file.name)[0])
            st.success("Analysis stored in vector database.")

            st.download_button("Download Analysis", analysis, file_name="resume_analysis.txt")

if __name__ == "__main__":
    main()