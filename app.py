"""
MIT License

Copyright (c) 2023 keanteng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# load libraries
import streamlit as st
import pandas as pd
import json
from backend.functions import *

# page setup
st.set_page_config(
    page_title = "Movie Recommendation With PaLM-2",
    page_icon = "🤖",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# sidebar
with st.sidebar:
    st.title("🍿 Movie Recommendation With PaLM-2")
    st.sidebar.caption("MIT License © 2023 keanteng")
    with st.expander("PaLM-2 API", expanded = True):
        api_toggle = st.toggle("Enable PaLM-2 API", value = False)
        api_input = st.text_input("PaLM-2 API Token", type="password", placeholder="Enter your PaLM-2 API token here")


# main page

## load data
movie_data = pd.read_excel("data.xlsx")

## configure API
if api_toggle:
    api_configure(api_key = api_input)
else:
    api_configure(api_key=PALM_TOKEN)
    
# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if is_valid_json(message["content"]) == True:
            temp = message["content"]
            temp = json_to_frame(message["content"])
            st.dataframe(temp, hide_index=True)
        else:
            st.markdown(message["content"])

# chatbot
prompt = st.chat_input("Enter your prompt here")

## workflow
model = load_llm()
movie_data = data_processing(movie_data)

with st.chat_message(name='AI', avatar="🎬"):
    st.write("Share your thoughts on a movie you like, and I'll recommend you a movie you might like!")
    
if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    to_llm = prompt_processing(prompt, movie_data)
    response = llm_agent(prompt=to_llm, model=model)
    response_df = json_to_frame(response)
    st.chat_message("assistant").dataframe(response_df, hide_index=True)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})