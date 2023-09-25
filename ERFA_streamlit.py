import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

import openai

openai.api_key = "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"


with open('document_embeddings.pkl', 'rb') as fp:
    document_embeddings = pickle.load(fp)

df = pd.read_csv('df.csv')

st.write(df)