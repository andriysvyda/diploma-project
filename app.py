import streamlit as st
from transformers import pipeline

st.title("Класифікатор твітів")
model = pipeline("text-classification", model="ukr-roberta-finetuned")

user_input = st.text_input("Введіть твіт:")
if user_input:
    result = model(user_input)[0]
    st.write(f"Тема: {result['label']} (Впевненість: {result['score']:.2f})")