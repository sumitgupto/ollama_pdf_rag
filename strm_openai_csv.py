"""
Streamlit application for CSV-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a CSV, process it,
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

#from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_milvus import Milvus
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

from typing import List, Tuple, Dict, Any, Optional

#from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import csv
csv.field_size_limit(10**6)

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
    embed_model_args = sys.argv[1] #text-embedding-3-large
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
    page_title="OPENAI CSV RAG using openAI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=True)
#hardcode model names
def extract_model_names():
    return llm_model_args


def create_vector_db(file_upload) -> Milvus:
    """
    Create a vector database from an uploaded CSV file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the CSV.

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
        loader = CSVLoader(file_path=path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
        print("Total documents : ", len(data))
        st.session_state["data"] = data

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size_args), chunk_overlap=int(chunk_overlap_args))
    chunks = text_splitter.split_documents(data)
    #print_chunks(chunks)
    print("Total Chunks : ", len(chunks))
    st.session_state["chunks"] = chunks
    logger.info("Document split into chunks")

    #embeddings = OllamaEmbeddings(model=embed_model_args, show_progress=True) #nomic-embed-text #mxbai-embed-large
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model= embed_model_args, dimensions=1536) #text-embedding-3-small
    
    #persist_directory_csv ='./chroma_db_csv'
    
    vector_db = Milvus.from_documents(  # or Zilliz.from_documents
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": "./milvus_db_csv.db"},
        drop_old=True,  # Drop the old Milvus collection if it exists
    )   
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return vector_db

def print_chunks(chunks) :
    for i in range(10) :
        print(f"\nChunk value in iteration {i} is : ", chunks[i])
    
    last_chunk = chunks[-10:]
    for j in range(len(last_chunk)) :
         print(f"\nChunk value in iteration {j} is : ", chunks[j])


def process_question(question: str, vector_db: Milvus, selected_model: str) -> str:
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
    
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    Look through the whole document before answering the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    The response should be specific and use statistics or numbers when possible.

    Assistant:"""

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    logger.info("got the prompt template")

    retriever = vector_db.as_retriever(search_kwargs={"k": 25})
    logger.info("got the retriever")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    logger.info("got the def_fomat")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | selected_model
        | StrOutputParser()
    )
    logger.info("got the chain")
    
    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a CSV file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the CSV.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"""Extracting all pages as images from file: {file_upload.name}""")

    import pdfkit
    import pandas as pd

    df = pd.read_csv(file_upload)
    html_table = df.to_html()

    options = {    'page-size': 'Letter',
    'margin-top': '0mm',
    'margin-right': '0mm',
    'margin-bottom': '0mm',
    'margin-left': '0mm'
    }

    pdf_name = "image.pdf"
    #pdfkit.configuration(wkhtmltopdf='D:/htmltopdf/wkhtmltopdf/bin')
    pdfkit.from_string(html_table, pdf_name, options=options)

    pdf_pages = []
    
    with pdfplumber.open(pdf_name) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Milvus]) -> None:
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
    st.subheader("🧠 Ollama CSV RAG playground", divider="gray", anchor=False)

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
        "Upload a CSV file ↓", type="csv", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)
        
        #first covert to pdf and then pass to this call
        #pdf_pages = extract_all_pages_as_images(file_upload)
        #st.session_state["pdf_pages"] = pdf_pages

    
        with col1:
            with st.container(height=50, border=True):
                st.write("total documents : ", len(st.session_state["data"]), "total chunks : ", len(st.session_state["chunks"]))


    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=300, border=True)

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
                            st.warning("Please upload a csv file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a csv file to begin chat...")


if __name__ == "__main__":
    main()
