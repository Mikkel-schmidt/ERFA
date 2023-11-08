import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tiktoken
from streamlit_extras.customize_running import center_running
center_running()

import openai


openai.api_key = st.secrets["apikey"]

if 'df' not in st.session_state:
    df = pd.read_csv('df_enc.csv')
    st.session_state['key'] = df

with open('document_embeddings.pkl', 'rb') as fp:
    document_embeddings = pickle.load(fp)

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

TEMPERATURE = 1
MAX_TOKENS = 1500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 3


def make_clickable(val, kilde):
    return f'<a target="_blank" href="{val}#page={page_number}">{kilde}</a>'



def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, previous_questions, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question+str(previous_questions), context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    st.write(f"Vigtigste {len(chosen_sections)} kilder:")
    #st.write("\n".join(chosen_sections))df_styled = df.head().style.format({'url': lambda x: make_clickable(x, df.loc[df['url'] == x]['Kilde'].values[0])})
    st.dataframe(df.iloc[chosen_sections_indexes])#.style.format({'url': lambda x: make_clickable(x, df.loc[df['url'] == x]['Kilde'].values[0])})['url'])
        
    return chosen_sections, chosen_sections_len

def get_response(instructions, previous_questions_and_answers, new_question, df, document_embeddings):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    prompt, section_lenght = construct_prompt(
        new_question,
        st.session_state.previous,
        document_embeddings,
        df
    )

    context= ""
    for article in prompt:
        context = context + article 

    messages.append({"role" : "user", "content":context})
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content, section_lenght


#INSTRUCTIONS = """Du skal komme med et forkert svar hver gang"""
INSTRUCTIONS = """Du er en rådgiver chatbot der kun kan svare ud fra den kontekst du er blevet tilført her. 
Hvis du ikke kan svare på spørgsmålet skal du svare 'Svaret er ikke i ERFA bladene, håndbogen eller Sikkerhedsstyrelsens guider.'"""
 

if 'previous' not in st.session_state:
    previous_questions_and_answers = []
    st.session_state['previous'] = previous_questions_and_answers


#prompt = st.chat_input('Indtast spørgsmål til ERFA-bladene, sikkerhedsstyrelsens guider eller håndbogen:', )
# if prompt:
#     c = st.container()
#     c.write(prompt)
#     errors = get_moderation(prompt)
#     if errors:
#         st.write(errors)
#     response, sections_tokens = get_response(INSTRUCTIONS, st.session_state.previous, prompt, df, document_embeddings)
#     c.write(response)

#     st.session_state.previous.append((prompt, response))
#     #st.write(df)
# st.write('Stil spørgsmål i søgefeltet i bunden')
# url = "https://forms.office.com/e/dtxKLNNWx8"
# st.write("Du kan komme med feedback [her](%s)" % url)

st.title("💬 FagBotten")
url = "https://forms.office.com/e/dtxKLNNWx8"
st.write("Du kan komme med feedback [her](%s)" % url)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Stil mig gerne et spørgsmål?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input('Indtast spørgsmål til ERFA-bladene, sikkerhedsstyrelsens guider eller håndbogen'):
     
    openai.api_key = st.secrets["apikey"]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response, sections_tokens = get_response(INSTRUCTIONS, st.session_state.messages, prompt, df, document_embeddings)
    msg = response#.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)

