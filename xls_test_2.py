# %%
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import os,glob
pwd=os.getcwd()

import sys
n = len(sys.argv)
print("Total arguments passed: ", n)

for i in range(1, n):
    llm_model_args = sys.argv[1] #gpt-4o-mini
    
print("llm_model from Args: ", llm_model_args)
llm=ChatOpenAI(model = llm_model_args) #c

st.title("CHAT WITH YOUR EXCEL FILE")
file=st.file_uploader("upload your file",type=["xlsx"])
if file:
    df=pd.read_excel(file)
    sdf=SmartDataframe ( 
        df,
        config={"llm":llm,"response_parser":StreamlitResponse}
    )

    input=st.text_area("ask your question here")
    if input is not None:
        btn=st.button("submit")
        if btn:
            response=sdf.chat(input)
            st.markdown(response)
            
                




