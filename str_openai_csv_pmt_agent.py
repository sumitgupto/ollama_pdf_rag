"""
streamlit run strm_openai.py text-embedding-3-small 1000 100 gpt-4o-mini
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import pandas as pd
import numpy as np
import ollama

#from langchain_community.document_loaders import UnstructuredPDFLoader
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

from langchain.chains import LLMChain

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from pandas import DataFrame

from langchain.callbacks import get_openai_callback

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


def create_vector_db(file_upload) -> Chroma:

    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        #loader = UnstructuredPDFLoader(path)
        df = pd.read_csv(path)
        df = df.replace(np.nan, "NA")

    logger.info("DF created and loaded")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return df


def process_question(question: str, df: DataFrame, selected_model: str) -> str:
    
    logger.info(f"""Processing question: {question} using model: {selected_model}""")

    llm = ChatOpenAI (temperature = 0, model_name = selected_model)

    template = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    You should interpret the columns of the dataframe as follows:

    1) Only answer questions related to the dataframe. For anything else, say you do not know.
    2) Each row of the dataframe corresponds to a git commit.
    3) The 1st column 'commit id' is the primary key or unique identifier for all rows
    4) The 2nd column 'Changes' shows the changes for the corresponding 'commit id' in the 1st column
    5) The 3rd column 'Comments' shows the code review comments for the corresponding 'commit id' in the 1st column and 'Changes' in the 2nd column
    9) If you do not know the answer, just say I don' know.

    You should use the tools below and the instructions above to answer the question posed of you:

    python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. 
    When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [python_repl_ast]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {question}
    {agent_scratchpad}
    """

    #prompt = PromptTemplate(template=template, input_variables=["question"])

    logger.info("Set the new template")

    agent = create_pandas_dataframe_agent (
        llm,
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        allow_dangerous_code=True,
        #number_of_head_rows = 5
    )

    #agent.agent.llm_chain.prompt.template = template

    response = count_tokens(agent, question)
    #response = agent.invoke(question)

    return response

def count_tokens(agent, question):
    with get_openai_callback() as cb:
        result = agent.invoke(question)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

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
    df = df.replace(np.nan, "NA")
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
    st.subheader("🧠 OPENAI CSV RAG playground", divider="gray", anchor=False)

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
        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=210, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

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
