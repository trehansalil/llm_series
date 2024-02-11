# Q&A Chatbot using LLMs

from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv() #Fetching the environment variable from .env

import streamlit as st
import os

# Function to load OpenAI model and get response

def get_openai_response(question):
    llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
    response = llm(question)
    
    return response

# Initializing Streamlit App

st.set_page_config(page_title="Q&A Demo")

st.header("Langchain Application")

input = st.text_input("Input: ", key="input")

response = get_openai_response(question=input)

submit = st.button("Ask the Question")

# If ask button is clicked

if submit:
    st.subheader('The response is')
    st.write(response)
    

