#!/usr/bin/env python
# coding: utf-8

import openai
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import streamlit as st
import json
from utils.process_df import *
from utils.get_prepoststicker import *
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

DATA_DIR     = os.path.join('data','raw','Medienmitteilungen Export DE 20230822.xlsx')
CRITERIA_DIR = os.path.join('data','processed','Medienmitteilungen Export DE 20230822- Kriterien der Konstruktivität updated.csv')
LOGO_PATH    = os.path.join('images','wwf_logo.jpg')

df  = pd.read_excel(DATA_DIR)
df2 = pd.read_csv(CRITERIA_DIR)

pdf = process_df(df)
df = pdf.skim_cols(df, 
                   keep_cols = ['Inhalt','Konstruktiv (1= eher konstruktiv   0 = eher nicht konstruktiv ','Hinweis'],
                   renamed_cols = ['content','label','reason'])
df = pdf.clean_df(df)

label_map = {
        1: "Texte, die als konstruktiv eingestuft werden, haben folgende Gründ:",
        0: "Texte, die als nicht konstruktiv/destruktiv eingestuft werden, haben folgende Gründ:"
}   

question = 'Als ässerst kritischer Umweltaktivist, dem der Erhalt der Umwelt am Herzen liegt, klassifizieren Sie den folgenden Text als konstruktiv oder destruktiv, indem Sie die oben genannten Beispielbegründungen zusammen mit Fragen und Lösungen im konstruktiven oder nicht konstruktiven/destruktiven Text verwenden. Bitte berücksichtigen Sie keine Kontaktdaten im Text. Erwähnen Sie zusammen mit der Textklassifizierung wichtige Punkte und entsprechende Gründe, warum der Text im folgenden JSON-Format als konstruktiv oder destruktiv klassifiziert wird: {"Klasse": "..", "Gründe dafür":  ["..",".."]}:'

presticker  = presticker_compute('',"v2", df, df2, label_map, question).get_presticker()
poststicker = poststicker_compute('',"v2").get_poststicker()

openai.api_key = os.environ.get('OPENAI_API_KEY')

state = st.session_state.get("state", [])

def ask_gpt(message):
    message_history = [{"role": "system", "content": "Sie sind ein unvoreingenommener und ässerst kritischer Umweltaktivist, der Texte schnell als destruktiv bis konstruktiv einstuft und sich um den Erhalt der Umwelt kümmert."},
                       {"role": "user", "content": presticker+message+poststicker}]
    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=message_history,
        temperature=st.session_state["temp"]
    )
    return response['choices'][0]['message']['content']



st.set_page_config(page_title='Rezensent von WWF-Artikeln',
                   layout='wide',
                   initial_sidebar_state='expanded',
                   page_icon=LOGO_PATH)


st.title("WWF :blue[Konstruktiver/Destruktiver Textklassifizierer]")


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "output_format" not in st.session_state:
    st.session_state["output_format"] = 'Table'

if "temp" not in st.session_state:
    st.session_state["temp"] = 0.2

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bitte geben Sie den Text ein, den Sie als konstruktiv oder nicht konstruktiv klassifizieren möchten"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ask_gpt(prompt)
        if st.session_state['output_format']=='Table':
            full_response = json.loads(full_response)
            full_response['Klasse'] = [full_response["Klasse"] for i in range(len(full_response["Gründe dafür"]))] 
            full_response = pd.DataFrame(full_response)
            st.dataframe(full_response, use_container_width=True)
        elif st.session_state['output_format']=='JSON':
            message_placeholder.markdown(full_response)
        elif st.session_state['output_format']=='Text':
            nl='\n'
            full_response = json.loads(full_response)
            cl     = 'Klasse: '+ full_response["Klasse"]
            reason = 'Gründe: '+' '.join(full_response["Gründe dafür"])
            st.markdown(cl)
            st.markdown('Gründe: ')
            for r in full_response["Gründe dafür"]:
                st.text(r)
        else:
            raise Exception('Unknown output_format')    
    
    st.session_state.messages = []

if st.button('Clear'):
    st.session_state.messages = []

st.sidebar.image(LOGO_PATH)
st.session_state["openai_model"] = str.strip(st.sidebar.text_input('Model', 'gpt-3.5-turbo-16k'))
st.session_state["temp"] = float(st.sidebar.slider('Model Parameters: Temperature/Randomness', 0.2, 0.8, 0.2))  
st.session_state["output_format"] = st.sidebar.radio("Select output format", ['Table', 'JSON', 'Text'])  

