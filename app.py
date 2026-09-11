import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
import joblib

st.set_page_config(page_title="Classificadores EMNIST", layout="wide")

st.title("Classificadores EMNIST - Binario, Digitos e Letras")
st.write(
    "Desenhe em cada canvas abaixo para acionar o classificador "
    "correspondente (treinado com SVM)."
)


# ------------------------------------------------------------
# Carregar os modelos treinados (uma unica vez, com cache)
# ------------------------------------------------------------
@st.cache_resource
def carregar_modelos():
    modelo_binario = joblib.load("modelos/modelo_binario.pkl")
    modelo_digitos = joblib.load("modelos/modelo_digitos.pkl")
    modelo_letras = joblib.load("modelos/modelo_letras.pkl")
    return modelo_binario, modelo_digitos, modelo_letras


modelo_binario, modelo_digitos, modelo_letras = carregar_modelos()


def corrigir_orientacao(img_array):
    img_array = np.rot90(img_array, k=-1)
    img_array = np.fliplr(img_array)
    return img_array


# ------------------------------------------------------------
# Funcao auxiliar: transforma o resultado do canvas no formato
# que o modelo espera (vetor de 784 posicoes, normalizado)
# ------------------------------------------------------------
def preprocessar(resultado_canvas):
    # resized_grayscale_array ja vem em 28x28
    img = resultado_canvas.resized_grayscale_array.astype(np.float32)
    img = corrigir_orientacao(img)
    img = img / 255.0
    img = img.reshape(1, -1)  # (1, 784)
    return img


# Mapeia o indice de volta para a letra (0->A, 1->B, ..., 4->E)
LETRAS_A_E = ["A", "B", "C", "D", "E"]

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# Canvas 1: Classificador Binario (eh o digito 5 ou nao?)
# ------------------------------------------------------------
with col1:
    st.subheader("Binario: eh o digito 5?")
    resultado_bin = st_mnist_canvas(key="canvas_binario")

    if resultado_bin.is_submitted:
        st.image(resultado_bin.resized_grayscale_array.astype("uint8"), caption="Imagem 28x28")
        entrada = preprocessar(resultado_bin)
        predicao = modelo_binario.predict(entrada)[0]

        if predicao:
            st.success("Resultado: E o digito 5! (Verdadeiro)")
        else:
            st.error("Resultado: NAO e o digito 5 (Falso)")

# ------------------------------------------------------------
# Canvas 2: Classificador multiclasse - digitos 1 a 5
# ------------------------------------------------------------
with col2:
    st.subheader("Digitos 1 a 5")
    resultado_dig = st_mnist_canvas(key="canvas_digitos")

    if resultado_dig.is_submitted:
        st.image(resultado_dig.resized_grayscale_array.astype("uint8"), caption="Imagem 28x28")
        entrada = preprocessar(resultado_dig)
        predicao = modelo_digitos.predict(entrada)[0]

        st.success(f"Digito previsto: {predicao}")

# ------------------------------------------------------------
# Canvas 3: Classificador multiclasse - letras A a E
# ------------------------------------------------------------
with col3:
    st.subheader("Letras A a E")
    resultado_letras = st_mnist_canvas(key="canvas_letras")

    if resultado_letras.is_submitted:
        st.image(resultado_letras.resized_grayscale_array.astype("uint8"), caption="Imagem 28x28")
        entrada = preprocessar(resultado_letras)
        predicao = modelo_letras.predict(entrada)[0]

        letra_prevista = LETRAS_A_E[predicao]
        st.success(f"Letra prevista: {letra_prevista}")

st.divider()
st.caption(
    "Se as predicoes vierem sempre erradas, o desenho pode precisar de "
    "ajuste de orientacao (rotacao/espelhamento) - o EMNIST guarda as "
    "imagens de um jeito diferente da orientacao natural de escrita. "
    "Veja a observacao no README."
)