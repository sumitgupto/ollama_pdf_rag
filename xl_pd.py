import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Define your desired Pandas DataFrame.
df=pd.read_excel(file_upload)



# Set up a parser + inject instructions into the prompt template.
parser = PandasDataFrameOutputParser(dataframe=df)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Here's an example of a column operation being performed.
df_query = "Retrieve the num_wings column."

# Set up the prompt.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
response = chain.invoke({"query": df_query})
print(response)
