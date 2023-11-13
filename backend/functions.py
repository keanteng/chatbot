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

import pandas as pd
import json
import streamlit as st
import google.generativeai as palm

# load token
try: 
    from backend.config import *
except ImportError:
    pass

# configure palm API
def api_configure(api_key = PALM_TOKEN):
    """
    Configure palm API with token.

    Args:
        token (string): PaLM-2 API token. Defaults to PALM_TOKEN.
    """
    palm.configure(api_key = PALM_TOKEN)

# load the large language model
def load_llm():
    """
    Load language model.

    Returns:
        Language Model: Text Bison 001
    """
    models = [
    m
    for m in palm.list_models()
    if "generateText" in m.supported_generation_methods
    ]
     # using text-bison-001
    model = models[0].name 
    
    return model

def data_processing(data):
    """
    Process data for the large language model.

    Args:
        data (DataFrame): Dataframe of the data.

    Returns:
        JSON output: JSON output of the data.
    """
    data_json = data.to_json(orient="records")
    return data_json

def is_valid_json(json_string):
    """
    Check if the string is a valid JSON.

    Args:
        json_string (String): String to be checked.

    Returns:
        Boolean: True if the string is a valid JSON, False otherwise.
    """
    try:
        json.loads(json_string)
        return True
    except ValueError:
        return False

# process and clean the prompt
def prompt_processing(user_instruct, json_data):
    """
    Process and clean the prompt.
    
    Args:
        user_instruct (string): Prompt from the user.
        json_data (JSON): JSON data from the large language model.
        
    Returns:
        string: Cleaned prompt.
    """
    instruct1 = """
    You will now only respoonse with JSON format.

    You will be feed with JSON data about shows on Netflix. For example, 
    """
    
    instruct2 ="""

    You must response with a JSON output consisting of a few movies will be watched by a person based on the data.

    Example1: Some Drame Movies
    Answer: [{"Movies":"The Social Dilemma"},{"Movies":"The Great Hack"},{"Movies":"The Big Hack"}]

    Please answer the following questions: """
    
    user_instruct = f'{user_instruct}'
    prompt = instruct1 + f'{json_data}' + instruct2 + user_instruct
    
    return prompt

def json_to_frame(output):
    """
    Convert json output to dataframe.

    Args:
        output (JSON): JSON output from the large language model.

    Returns:
        DataFrame: DataFrame of the JSON output.
    """
    json_output = json.loads(output)
    df_output = pd.DataFrame(json_output)
    return df_output

# accessing the llm with prompt
def llm_agent(prompt, model):
    """
    Accessing the large language model with prompt.

    Args:
        prompt (string): Prompt for the language model.
        model (model): Language model.Text Bison 001

    Returns:
        String: Output from the large language model.
    """
    completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
    )
    
    return completion.result

# testing
#api_configure(api_key=PALM_TOKEN)
#model = load_llm()
#prompt = prompt_processing("Some actions movies")
#output = llm_agent(prompt, model)
#df_output = json_to_frame(output)
#print(df_output)
#print(output)

#df = pd.read_excel('data.xlsx')
#a = prompt_processing("Some actions movies", data_processing(df))
#print(a)

