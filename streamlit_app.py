import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from model.inference import CBOWPredictor

st.set_page_config(page_title="CBOW Predictor", page_icon="🧠", layout="centered")
st.title("CBOW: Prediccion de Palabra Central")
st.caption("Ingresa 2 palabras a la izquierda y 2 a la derecha")

artifact_path = st.text_input("Ruta del modelo (.npz)", value="outputs/model/model_artifact.npz")

if "predictor" not in st.session_state:
    st.session_state.predictor = None

if st.button("Cargar modelo"):
    try:
        st.session_state.predictor = CBOWPredictor(artifact_path)
        st.success("Modelo cargado")
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo: {exc}")

left_col, right_col = st.columns(2)
with left_col:
    left2 = st.text_input("Palabra izquierda 2", value="").strip().lower()
    left1 = st.text_input("Palabra izquierda 1", value="").strip().lower()
with right_col:
    right1 = st.text_input("Palabra derecha 1", value="").strip().lower()
    right2 = st.text_input("Palabra derecha 2", value="").strip().lower()

if st.button("Predecir palabra central"):
    if st.session_state.predictor is None:
        st.warning("Primero carga un modelo")
    else:
        context = [left2, left1, right1, right2]
        if any(not token for token in context):
            st.warning("Completa las 4 palabras")
        else:
            try:
                prediction, top_k = st.session_state.predictor.predict(context, top_k=5)
                st.subheader(f"Prediccion: {prediction}")
                st.write("Top-5 probabilidades")
                st.table({"palabra": [item[0] for item in top_k], "probabilidad": [item[1] for item in top_k]})
            except Exception as exc:
                st.error(str(exc))
