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

import pandas
import json
import streamlit as st
import google.generativeai as palm

# load token
try: 
    from backend.config import *
except ImportError:
    pass

# configure palm API
def api_configure(token = PALM_TOKEN):
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

# process and clean the prompt
def prompt_processing(num):
    """
    Process and clean the prompt.
    
    Args:
        num (int): Number of movies to be recommended.
        
    Returns:
        string: Cleaned prompt.
    """
    text = """
    You will now only respoonse with JSON format.

    You will be feed with JSON data about shows on Netflix. For example,
    {"Netflix Shows":"100 humans","Genre":"Science","Type":"Documentary","Language":"English"}

    You will then response with a JSON output consisting of a few movies will be watched by a person.

    Example1: Number of movies: 3
    Answer: [Jujutsu Kaisen, My Happy Marriage, A Man Calling Otto]

    Please answer the following questions: """
    temp = f'Number of moviews: {num}'
    prompt = text + temp
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
        _type_: _description_
    """
    completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    # The maximum length of the response
    max_output_tokens=800,
    )
    
    return completion.result