import streamlit as st
import logging
import os
import tempfile
import shutil
import ollama

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

from typing import List, Tuple, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
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

#read dotenv
from dotenv import load_dotenv, find_dotenv
founddotenv = load_dotenv(find_dotenv(), override=True) 
logger.info("Found .env: %s", founddotenv)

# Streamlit page configuration
st.set_page_config(
    page_title="OLLAMA-CHROMA CSV RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def get_file_type(file_name) :
    # this will return a tuple of root and extension
    split_tup = os.path.splitext(file_name)
    # extract the file name and extension
    logger.info("File name is %s", split_tup)
    file_extension = split_tup[1]
    #logger.info("File type is %s", file_extension)
    return file_extension

@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],) -> Tuple[str, ...]:
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names

def delete_collection_if_exists(vector_store, embeddings):
    logger.info("found collection to be deleted : %s", vector_store._collection.count())
    vector_store.delete_collection()
    #Initialize the collection again and return
    vector_store = initialize_chroma_db(embeddings)
    return vector_store

def load_csv_data(path) :
    #loader = UnstructuredPDFLoader(path)
    loader = CSVLoader(file_path=path, encoding="utf-8", csv_args={
        "delimiter": ",",
        #"fieldnames": ["Commit ID", "Changes", "Comments"]
        }, 
        #source_column="Commit ID"
        )
    data = loader.load()
    logger.info("Total documents CSV : %s", len(data))
    st.session_state["data"] = data
    return data

def load_pdf_data(path) :
    loader = PyPDFLoader(path)
    data = loader.load_and_split()
    logger.info("Total documents PDF : %s", len(data))
    st.session_state["data"] = data
    return data

def initialize_chroma_db(embeddings) :
    vector_store = Chroma(
        collection_name="col_chroma_langchain_db_multi",
        embedding_function = embeddings,
        persist_directory="./dir_chroma_langchain_db_multi",
        # Where to save data locally, remove if not neccesary
    )
    logger.info("vector_store count at Initialization = %s", vector_store._collection.count())
    return vector_store

def get_chunk_list(chunks, count_max) :
    quotient = len(chunks) // count_max
    remainder = len(chunks) % count_max
    if remainder > 0:
        total_chunks = quotient + 1
    else :
        total_chunks = quotient
    #logger.info("Number of Chunk Groups : %s", total_chunks)

    chunk_list = []
    for i in range(total_chunks):
        chunk_name = f"chunk-{i}"
        chunk_list.append(chunk_name)
        #logger.info("Chunk name : %s", chunk_name)
    return chunk_list

def create_vector_db(file_upload) -> Chroma:
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
    
    logger.info(f"File saved to temporary path: {path}")
    
    file_type = get_file_type(file_upload.name)
    logger.info("file Type is %s", file_type)

    if file_type == ".csv":
        data = load_csv_data(path)
    elif file_type == ".pdf":
        data = load_pdf_data(path)
    else:
        logger.error("File format NOT supported")
    
    logger.info("chunk_size = %s", st.session_state['chunk_size'])
    logger.info("chunk_overlap = %s", st.session_state['chunk_overlap'])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(st.session_state['chunk_size']), chunk_overlap=int(st.session_state['chunk_overlap']))
    chunks = text_splitter.split_documents(data)
    logger.info("Total Chunks : %s", len(chunks))
    st.session_state["chunks"] = chunks
    logger.info("Document split into chunks")

    ##code to batch
    count_max = 2000
    chunk_list = get_chunk_list(chunks, count_max)
    logger.info("Total Chunk List : %s", len(chunk_list))

    ##end of batch logic
    logger.info("embed_name = %s", st.session_state['embed_name'])
    embeddings = OllamaEmbeddings(model=st.session_state['embed_name'], show_progress=True) #nomic-embed-text #mxbai-embed-large

    #Initiaze Chroma DB
    vector_db = initialize_chroma_db(embeddings)

    #delete existing collections
    if (vector_db._collection.count()) > 0:
        vector_db = delete_collection_if_exists(vector_db, embeddings)

    #insert new data
    vector_db = insert_into_chroma(vector_db, chunks, chunk_list, count_max, embeddings)
    logger.info("Total Collection count in DB : %s", vector_db._collection.count())
    st.session_state["total_db_records"] = vector_db._collection.count()

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return vector_db

def insert_into_chroma(vectorstore, chunks, chunk_list, count_max, embeddings):
    remaining = len(chunks)
    for name in chunk_list:
        if remaining > count_max :
            name = chunks[:count_max]
            remaining = remaining - count_max
            logger.info("Chunk Size : %s", len(name))
            vectorstore = send_chunks_to_chroma_Db(vectorstore, name, embeddings)
            logger.info("sent chunks to DB : %s", len(name))
        else :
            name = chunks[:remaining]
            logger.info("Last Chunk Size : %s", len(name))
            vectorstore = send_chunks_to_chroma_Db(vectorstore, name, embeddings)
            logger.info("sent last chunks to DB : %s", len(name))
    return vectorstore

def send_chunks_to_chroma_Db (vectorstore, subchunk, embeddings) :   
    vectorstore.add_documents(subchunk)
    logger.info(f"Added {vectorstore._collection.count()} chunks to vectorstore")
    return vectorstore

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:

    logger.info(f"""Processing question: {question} using model: {selected_model}""")
    
    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using 
    fact based and statistical information derived from the information provided.
    Use the following pieces of information to provide a concise answer 
    to the question enclosed in <question> tags.
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

    retriever = vector_db.as_retriever(search_kwargs={"k": int(st.session_state['k_args'])})
    logger.info("k_args = %s", st.session_state['k_args'])
    logger.info("got the retriever")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    logger.info("got the def_fomat")

    llm = ChatOllama(model=selected_model, temperature=0)
    logger.info("got the llm")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("got the chain")
    
    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def clear_text():
        st.session_state["1"] = None
        st.session_state["2"] = None
        st.session_state["3"] = None
        st.session_state["4"] = None

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    if vector_db is not None:
        logger.info("Deleting vector DB as vector_db is not None")
        update_key()
        vector_db.delete_collection()
        st.session_state["vector_db"] = None

        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("chunk_size", None)
        st.session_state.pop("chunk_overlap", None)
        st.session_state.pop("embed_name", None)
        st.session_state.pop("k_args", None)
        st.session_state.pop("total_db_records", None)

        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.session_state["vector_db"] = None
        st.error("No vector database found to delete and st.session_state[vector_db] set to None")
        logger.warning("Attempted to delete vector DB, but none was found")

def update_key():
    st.session_state.uploader_key += 1
    st.cache_data.clear()

def check_mandatory_values():
    flag = False
    if ("k_args" in st.session_state) and ("chunk_size" in st.session_state) and ("chunk_overlap" in st.session_state) and ("embed_name" in st.session_state) :
        flag = True
    return flag

def main() -> None:

    st.subheader("🤖 Ollama CSV Chroma RAG playground", divider="blue", anchor=False)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    with col1:
            with st.container(height=260, border=True):
                col3, col4 = st.columns([1,1])
                with col3:
                    with st.container(height=260, border=True):
                        chunk_size = st.text_input(label='Enter chunk size', key="1")
                        chunk_overlap = st.text_input(label='Enter chunk overlap', key="2")

                with col4:
                    with st.container(height=260, border=True):
                        k_args = st.text_input(label='Enter K docs to retrieve', key="3")
                        embed_name = st.text_input(label='Enter Embedding Model Name', key="4")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    file_upload = col1.file_uploader(
        "Upload a CSV or PDF file ↓", 
        type=['csv', 'pdf'], 
        accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}"
    )
 
    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["chunk_size"] = chunk_size
            st.session_state["chunk_overlap"] = chunk_overlap
            st.session_state["k_args"] = k_args
            st.session_state["embed_name"] = embed_name
            st.session_state["vector_db"] = create_vector_db(file_upload)
        
        with col1:
            with st.container(height=50, border=True):
                st.write("total documents : ", len(st.session_state["data"]), 
                        "total chunks : ", len(st.session_state["chunks"]),
                        "total doc count in DB : ", st.session_state["total_db_records"]
                        )

    delete_collection = col1.button("⚠️ Delete collection", type="secondary", on_click=clear_text)

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
                            st.session_state["chunk_size"] = chunk_size
                            st.session_state["chunk_overlap"] = chunk_overlap
                            st.session_state["k_args"] = k_args
                            st.session_state["embed_name"] = embed_name
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a csv or pdf file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a csv or pdf file to begin chat...")


if __name__ == "__main__":
    main()
