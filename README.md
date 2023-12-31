# Streamlit Chatbot <!-- omit in toc -->

![Static Badge](https://img.shields.io/badge/license-MIT-blue)
![Static Badge](https://img.shields.io/badge/python-3.11-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A chatbot capable of recommending Netflix movies of shows created using the Streamlit framework using `PaLM-2` API.

**Table of Contents:**
- [About](#about)
  - [Purpose](#purpose)
- [Using the Repository](#using-the-repository)
- [Work In Progress](#work-in-progress)

## About

### Purpose
Allow large language model to read input data and make recommendation from the data. The data can be of jobs, movies, songs or texts related. 

## Using the Repository
To clone this repository to your local machine:

```py
git clone https://github.com/keanteng/chatbot
```

After cloning this repository, you can run the application locally:

```py
py -m streamlt run app.py
```

If you are using a virtual environment `.venv`, you can install the dependencies:

```py
py -m pip install -r requirements.txt
```

## Work In Progress
1. Adding Retrieval Augmented Generation (RAG) feature to enhance model capability
    - Currently `llama-index` only supports OpenAI use cases. Free experimental features from PaLM-2 is not available.
2. Refine front-end experience for a better chatbot interaction