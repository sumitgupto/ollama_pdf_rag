"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.

streamlit run strm_openai.py text-embedding-3-small 1000 100 gpt-4o-mini
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
#import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

import sys
# total arguments
n = len(sys.argv)
#logger.info(f"Extracted model names: {model_names}")
logger.info("Total arguments passed: %s", n)

for i in range(1, n):
    embed_model_args = sys.argv[1]
    chunk_size_args = sys.argv[2]
    chunk_overlap_args = sys.argv[3]
    llm_model_args = sys.argv[4] #gpt-4o-mini
    
#print("llm_model from Args:", llm_model_args)
logger.info("embed_model from Args: %s", embed_model_args)
logger.info("chunk_size from Args: %s", chunk_size_args)
logger.info("chunk_overlap from Args: %s", chunk_overlap_args)
logger.info("llm_model from Args: %s", llm_model_args)

#read dotenv
from dotenv import load_dotenv, find_dotenv
founddotenv = load_dotenv(find_dotenv(), override=True) 
logger.info("Found .env: %s", founddotenv)


# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG using openAI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=True)
#hardcode model names
def extract_model_names():
    return llm_model_args


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the pdf.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        #loader = UnstructuredPDFLoader(path)
        loader = UnstructuredPDFLoader(file_path=path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size_args), chunk_overlap=int(chunk_overlap_args))
    chunks = text_splitter.split_documents(data)
    logger.info(f"Document split into {len(chunks)} chunks")

    #embeddings = OllamaEmbeddings(model=embed_model_args, show_progress=True) #nomic-embed-text #mxbai-embed-large
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model= embed_model_args, dimensions=1536) #text-embedding-3-small
    persist_directory_pdf ='./chroma_db_pdf'
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory_pdf, 
        collection_name="myRAG-pdf"
        )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"""Processing question: {question} using model: {selected_model}""")
    #llm = ChatOllama(model=selected_model, temperature=0)
    llm = ChatOpenAI(model=selected_model, temperature=0)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant and you understand git commit data.
        You shall review documents containing git commits. The first column is "commit id", 2nd column is "Changes" 
        and 3rd column is "Comments".
        you should be able to count git commits and find similarities and dissimilarities between 2 git commits
        Your task is to generate 3 different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a pdf file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the pdf.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"""Extracting all pages as images from file: {file_upload.name}""")

    pdf_pages = []
    
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    #models_info = ollama.list()
    #available_models = extract_model_names(models_info)
    available_models = extract_model_names()

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)
        
        #first covert to pdf and then pass to this call
        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a pdf file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a pdf file to begin chat...")


if __name__ == "__main__":
    main()
