import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
import joblib

st.set_page_config(page_title="Calibrar orientacao EMNIST", layout="wide")

st.title("Calibrador de orientacao")
st.write(
    "Desenhe o digito 5 no canvas abaixo. O app vai testar as 4 "
    "combinacoes possiveis de rotacao/espelhamento e mostrar a "
    "predicao de cada uma - assim voce descobre qual e a correta."
)


@st.cache_resource
def carregar_modelo():
    return joblib.load("modelos/modelo_binario.pkl")


modelo = carregar_modelo()

resultado = st_mnist_canvas(key="canvas_calibracao")

# As 4 transformacoes possiveis
TRANSFORMACOES = {
    "Original (sem alteracao)": lambda img: img,
    "Rotacionar 90 graus": lambda img: np.rot90(img, k=-1),
    "Espelhar horizontal": lambda img: np.fliplr(img),
    "Rotacionar 90 + espelhar": lambda img: np.fliplr(np.rot90(img, k=-1)),
}

if resultado.is_submitted:
    img_original = resultado.resized_grayscale_array.astype(np.float32)

    st.subheader("Sua imagem original (como o canvas gerou)")
    st.image(img_original.astype(np.uint8), width=150)

    st.subheader("Testando as 4 orientacoes:")
    cols = st.columns(4)

    for col, (nome, transformacao) in zip(cols, TRANSFORMACOES.items()):
        with col:
            img_transformada = transformacao(img_original.copy())

            entrada = (img_transformada / 255.0).reshape(1, -1)
            predicao = modelo.predict(entrada)[0]

            st.image(img_transformada.astype(np.uint8), caption=nome, width=140)
            if predicao:
                st.success("Previu: E o 5!")
            else:
                st.error("Previu: NAO e o 5")

    st.info(
        "Anote qual das 4 opcoes acertou (previu 'E o 5!' quando voce "
        "desenhou um 5 de verdade). Use essa mesma transformacao na "
        "funcao preprocessar() do app.py final."
    )