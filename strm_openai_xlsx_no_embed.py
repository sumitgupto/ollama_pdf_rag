import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
#import ollama


from langchain_community.vectorstores import Chroma
#from langchain_community.chat_models import ChatOllama

from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate

from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
from pandas import DataFrame
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse


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
    llm_model_args = sys.argv[1] #gpt-4o-mini
    
#print("llm_model from Args:", llm_model_args
logger.info("llm_model from Args: %s", llm_model_args)

#read dotenv
from dotenv import load_dotenv, find_dotenv
founddotenv = load_dotenv(find_dotenv(), override=True) 
logger.info("Found .env: %s", founddotenv)


# Streamlit page configuration
st.set_page_config(
    page_title="OpenAI XLSX Chatbot",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

@st.cache_resource(show_spinner=True)
#hardcode model names
def extract_model_names():
    return llm_model_args


def create_vector_db(file_upload) -> DataFrame:

    logger.info(f"Creating vector DB from file upload: {file_upload.name}")

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")

    #llm = ChatOpenAI(model=llm_model_args, temperature=0)

    df=pd.read_excel(file_upload)

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return df


def process_question(question: str, sdf: DataFrame, selected_model: str) -> str:

    logger.info(f"""Processing question: {question} using model: {selected_model}""")

    parser = PandasDataFrameOutputParser(DataFrame=sdf)

    # Set up the prompt.
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = ChatOpenAI(model=selected_model, temperature=0)

    chain = prompt | model | parser
    response = chain.invoke({"query": question})

    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:

    logger.info(f"""Extracting all pages as images from file: {file_upload.name}""")

    import pdfkit
    import pandas as pd

    # read by default 1st sheet of an excel file
    try: 
        df = pd.read_excel(file_upload)
        html_table = df.to_html()
    except:
        logger.error("Error Occurred while reading excel sheet")

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

    st.subheader("🧠 OpenAI XLSX RAG playground", divider="gray", anchor=False)

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
        "Upload a xlsx file ↓", type="xlsx", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)
            logger.info("st.session_state['vector_db'] updated in session")

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
                            response = process_question (
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a xlsx file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a xlsx file to begin chat...")


if __name__ == "__main__":
    main()
