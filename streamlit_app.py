import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from model.inference import LMPredictor

st.set_page_config(page_title="LLM (No-Transformer)", page_icon="🤖", layout="centered")
st.title("🤖 Generador de Texto MLP")
st.caption("Arquitectura Autoregresiva basada en BPE y Multilayer Perceptron")

col1, col2 = st.columns(2)
with col1:
    artifact_path = st.text_input("Ruta de Pesos (.npz)", value="outputs/model/lm_weights.npz")
with col2:
    tokenizer_path = st.text_input("Ruta de Tokenizador (.json)", value="outputs/model/bpe_tokenizer.json")

if "predictor" not in st.session_state:
    st.session_state.predictor = None

if st.button("Cargar modelo"):
    if not os.path.exists(artifact_path) or not os.path.exists(tokenizer_path):
        st.error("No se encuentran los archivos. ¿Ya corriste `python test_train.py`?")
    else:
        try:
            # El predictor infiere automáticamente el context_size de los pesos
            st.session_state.predictor = LMPredictor(artifact_path, tokenizer_path)
            st.success("¡Modelo cargado exitosamente!")
        except Exception as exc:
            st.error(f"Error al cargar el modelo: {exc}")

st.divider()
st.subheader("Generación de Texto")

prompt = st.text_input("Escribe el inicio de la frase:", value="el perro")

col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("Predecir SIGUIENTE palabra"):
    if st.session_state.predictor is None:
        st.warning("Primero carga un modelo.")
    elif not prompt.strip():
        st.warning("Escribe algo de texto.")
    else:
        try:
            next_word, top_k = st.session_state.predictor.generate_next_token(prompt, top_k=5)
            st.success(f"Palabra generada: **{next_word}**")
            
            st.write("Top-5 alternativas evaluadas por la capa Softmax:")
            st.table({"Palabra": [item[0] for item in top_k], "Probabilidad": [round(item[1], 4) for item in top_k]})
        except Exception as exc:
            st.error(str(exc))

if col_btn2.button("Autocompletar frase entera"):
    if st.session_state.predictor is None:
        st.warning("Primero carga un modelo.")
    elif not prompt.strip():
        st.warning("Escribe algo de texto.")
    else:
        try:
            with st.spinner("Generando iterativamente..."):
                full_text = st.session_state.predictor.generate_text(prompt, max_tokens=6)
            st.success(f"Texto autocompletado: **{full_text}**")
        except Exception as exc:
            st.error(str(exc))
