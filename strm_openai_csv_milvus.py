# %%
from dotenv import load_dotenv, find_dotenv
founddotenv = load_dotenv(find_dotenv(), override=True) 
print("Found .env: %s", founddotenv)

# %%
from langchain_community.document_loaders.csv_loader import CSVLoader
path = "./pull_requests_summary.csv"

import csv
csv.field_size_limit(10**6)

loader = CSVLoader(file_path=path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
chunks = text_splitter.split_documents(data)
print("Chunks : ",len(chunks))

# %%
from langchain_openai import OpenAIEmbeddings

#embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(model= "text-embedding-3-small", dimensions=1536) #text-embedding-3-small

# %%
from langchain_milvus import Milvus
vectorstore = Milvus.from_documents(  # or Zilliz.from_documents
    documents=chunks,
    embedding=embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    #drop_old=True,  # Drop the old Milvus collection if it exists
)

# %%
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db_csv", 
        collection_name="myRAG-CSV"
        )

# %%
query = "How many commit id are present?"
response = vectorstore.similarity_search(query, k=1)
print(response)

# %%
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "How many commit id are present?"

print("Asking the question : ", query)

res = rag_chain.invoke(query)
print(res)



